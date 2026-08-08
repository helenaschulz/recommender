"""Precompute everything the demo app serves from, so nothing slow happens at query time.

    python scripts/build_app_assets.py

Writes to the gitignored ``artifacts/app/``. Run it once after cloning; re-run it only if
``data/`` or the asset layout changes (:data:`recommender.demo.ASSET_VERSION`).

**This script fits on the FULL interaction matrix, and that is not leakage.** Every model
in ``docs/RESULTS.md`` fits on ``split.train``, because a metric measured on data the model
has already seen is worthless. This script computes no metric. It builds the *serving*
path, where withholding a reader's interactions would make the product worse for no reason
at all. The two are different jobs and the distinction is written here rather than left for
a reviewer to wonder about: nothing produced by this script may ever be quoted as a result.

**The item is a work, matching the published table** (M12, ledger L49). Editions are merged
before fitting, which is where the project's largest single accuracy gain came from — and,
more visibly for a demo, what turns item-item's *Harry Potter* neighbourhood from two
obscure books into the four sequels in order (L53). Serving the demo on the ISBN basis while
the case argues for works would have been an odd thing to put in front of a panel. It also
makes serving-time deduplication (L45) unnecessary here: the collapse already happened.

What lands on disk, and why each piece is needed:

- ``factors.npy`` / ``item_ids.npy`` / ``item_support.npy`` — L2-normalized ALS item
  factors, pre-normalized here so a query is a single mat-vec, plus the support counts the
  L34 floor filters on.
- ``readers.npz`` — the interaction matrix transposed to (item x user), so "how many
  readers of X also read Y" is an intersection of two sorted index slices.
- ``lookup_vectors.npy`` / ``lookup_ids.npy`` / ``lookup_support.npy`` — the **uncentered**
  sentence embeddings the free-text box searches (ledger L37), plus the support counts that
  keep the box from offering an anchor the engine would refuse to answer for.
- ``books.parquet`` — one row per work: the title, author, year, series and cover URL of
  its most-interacted edition.

The sentence encoder is also warmed here (one throwaway query), so the first real query in
front of an audience does not trigger a model download.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

from recommender.data import build_interactions, cluster_works, load, work_level_catalog
from recommender.demo import ASSET_VERSION, assets_dir
from recommender.models.als import ALSRecommender
from recommender.models.embeddings import DEFAULT_MODEL, EmbeddingRecommender

#: Columns the app displays. Everything else is dropped so the metadata table stays small.
DISPLAY_COLUMNS = ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "series", "Image-URL-M"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--similar-min-support", type=int, default=20, help="candidates; ledger L34")
    parser.add_argument("--anchor-min-support", type=int, default=50, help="anchors; ledger L63/L65, M14.2")
    parser.add_argument(
        "--isbn-work-key",
        action="store_true",
        help="build on the published M11 work key instead of the M14.4-fixed one (L64)",
    )
    parser.add_argument("--skip-warmup", action="store_true", help="do not load the sentence encoder")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    out = assets_dir()
    out.mkdir(parents=True, exist_ok=True)

    raw = load()
    # One item per *work*, matching the published table (M12). There is no holdout here --
    # this is serving -- so the canonical per-work text is chosen on everything, which is
    # the right thing for a product and would be leakage in an evaluation.
    #
    # **The serving key carries the M14.4 punctuation fix; the published table does not.**
    # Helena's decision, on the priced options: the fix merges 1,198 works with 0 wrong
    # merges in a 30-cluster audit, and the demo is where it is visible -- without it
    # "Bridget Jones's Diary" answers with the same book at ranks 1 and 2. Re-basing the
    # whole M12 table for +0.8% on one row was not worth the day it would cost with the
    # deck unwritten. The divergence is deliberate, priced in ledger L64, and recorded in
    # meta.json so nobody has to infer it from a work count.
    works = raw.works if args.isbn_work_key else cluster_works(raw.books, normalize_punctuation=True)
    catalog = work_level_catalog(raw, works)
    # The whole matrix on purpose -- see the module docstring. Binarized, matching the
    # pinned signal decision: an interaction is evidence whether or not it carries a grade.
    interactions = build_interactions(catalog.ratings, weights="binary")
    print(
        f"catalogue: {len(raw.books):,} ISBNs -> {len(catalog.books):,} works, "
        f"{len(catalog.ratings):,} interactions, matrix {interactions.n_users:,} x "
        f"{interactions.n_items:,}  [{time.perf_counter() - started:.0f}s]",
        flush=True,
    )

    step = time.perf_counter()
    als = ALSRecommender(
        factors=args.factors,
        alpha=args.alpha,
        regularization=args.regularization,
        similar_min_support=args.similar_min_support,
    ).fit(interactions, catalog, ratings=catalog.ratings)
    factors = np.asarray(als.item_factors, dtype=np.float32)
    norms = np.linalg.norm(factors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    np.save(out / "factors.npy", factors / norms)
    np.save(out / "item_ids.npy", interactions.item_ids)
    np.save(out / "item_support.npy", interactions.item_popularity)
    print(f"ALS fitted and saved: {factors.shape[0]:,} x {factors.shape[1]}  [{time.perf_counter() - step:.0f}s]",
          flush=True)

    step = time.perf_counter()
    # (item x user): one row per book listing its readers, so a co-reader count is an
    # intersection of two sorted slices rather than a sparse product.
    sp.save_npz(out / "readers.npz", sp.csr_matrix(interactions.matrix.T))
    print(f"reader index saved  [{time.perf_counter() - step:.0f}s]", flush=True)

    step = time.perf_counter()
    embeddings = EmbeddingRecommender().fit(interactions, catalog)
    np.save(out / "lookup_vectors.npy", np.asarray(embeddings.lookup_vectors, dtype=np.float32))
    np.save(out / "lookup_ids.npy", embeddings.item_ids)
    # Support aligned to the lookup rows, so the app can apply the same L34 floor to the
    # *input* path without joining two id spaces at query time. Books that appear in
    # Books.csv but in no interaction get 0 and are therefore never offered as an anchor.
    support_by_isbn = dict(zip(interactions.item_ids.tolist(), interactions.item_popularity.tolist(), strict=True))
    np.save(
        out / "lookup_support.npy",
        np.array([support_by_isbn.get(str(isbn), 0) for isbn in embeddings.item_ids], dtype=np.int64),
    )
    print(
        f"lookup vectors saved: {embeddings.lookup_vectors.shape}  "
        f"(cache {embeddings.cache_path})  [{time.perf_counter() - step:.0f}s]",
        flush=True,
    )

    step = time.perf_counter()
    # The ISBN column already holds work ids at this point, and one row per work is what
    # the app displays: title, author and cover of the work's most-interacted edition.
    books = catalog.books.reindex(columns=DISPLAY_COLUMNS).fillna("")
    books.to_parquet(out / "books.parquet", index=False)
    print(f"metadata saved: {len(books):,} works  [{time.perf_counter() - step:.0f}s]", flush=True)

    if not args.skip_warmup:
        step = time.perf_counter()
        from recommender.models.embeddings import sentence_transformer_encoder

        # One throwaway query, so the model is in the local cache before a live demo asks
        # for it. This is the only place the app's stack ever touches the network.
        sentence_transformer_encoder(DEFAULT_MODEL)(["harry potter stein"])
        print(f"sentence encoder warmed  [{time.perf_counter() - step:.0f}s]", flush=True)

    (out / "meta.json").write_text(
        json.dumps(
            {
                "asset_version": ASSET_VERSION,
                "similar_min_support": args.similar_min_support,
                "anchor_min_support": args.anchor_min_support,
                "encoder_model": DEFAULT_MODEL,
                "als": als.describe_params(),
                "fitted_on": "the full work-keyed interaction matrix (serving, not evaluation)",
                "item_level": "work",
                "work_key": "M11" if args.isbn_work_key else "M11 + M14.4 punctuation normalization (L64)",
                "n_items": int(interactions.n_items),
                "n_works": int(len(books)),
                "n_isbns": int(len(raw.books)),
            },
            indent=1,
        )
    )

    total = sum(path.stat().st_size for path in out.iterdir()) / 1e6
    print(f"\nassets in {out}  ({total:,.0f} MB)  build time {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
