"""Serving-time work dedup (M11.5) and the coverage counting-basis fix (M11.6).

    python scripts/analyze_dedup.py
    python scripts/analyze_dedup.py --models item-item tfidf --limit-users 3000

Every model is scored **twice**: once at ``k``, which is exactly the call the comparison
table made, and once at ``k * oversample``, which is what dedup collapses. Slicing the
first ten columns off the wide pass would have been cheaper and is *almost* the same
thing — but not quite: with tied scores, ``argpartition`` selects a different set of ties
at k=10 than at k=100, and item-item's HitRate moves 0.0546 -> 0.0543 on that alone. The
"before" column has to be the number the ledger already reports, so it is measured the
way the ledger measured it. The metric functions come from :mod:`recommender.eval` rather
than being re-derived here, for the same reason.

Two questions, kept apart:

- **M11.5** — what does collapsing editions at serving time do to the duplicate rate
  ledger L31 measured (31.6% of TF-IDF's slots), to the three metrics, and to the gallery?
- **M11.6** — Coverage@10 was reported on two different denominators. L24 counts distinct
  recommended ISBNs that exist in ``Books.csv``; the complementarity paragraph and L32
  counted *all* distinct recommended ISBNs, including the ones with no catalogue row.
  Both bases are printed side by side so the ledger can be corrected against a number
  rather than an argument.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from recommender.data import CATALOG_SIZE, build_interactions, load
from recommender.eval import catalog_coverage_at_k, hit_rate_at_k, novelty_at_k
from recommender.gallery import build_gallery, render_markdown
from recommender.models import build_model, fit_model
from recommender.serving import WorkDeduped, duplicate_slot_rate
from recommender.split import make_split

DEFAULT_MODELS = ["item-item", "tfidf", "embeddings", "als"]


def summarize(label, recommended, holdout, train, works, catalog_isbns, user_ids) -> dict:
    flat = [isbn for isbn in recommended.ravel().tolist() if isbn is not None]
    distinct_all = set(flat)
    distinct_catalog = distinct_all & catalog_isbns
    distinct_works = set(works.of(sorted(distinct_catalog)).tolist()) if distinct_catalog else set()
    popularity = dict(zip(train.item_ids.tolist(), train.item_popularity.tolist(), strict=True))
    dup = duplicate_slot_rate(recommended, user_ids, train, works)
    filled = len(flat)
    return {
        "label": label,
        "hit_rate": hit_rate_at_k(recommended, holdout),
        "coverage_catalog": catalog_coverage_at_k(recommended, catalog_isbns),
        "coverage_all_isbns": len(distinct_all) / CATALOG_SIZE,
        "coverage_works": len(distinct_works) / works.n_works,
        "novelty": novelty_at_k(recommended, popularity, int(train.item_popularity.sum())),
        "distinct_all": len(distinct_all),
        "distinct_catalog": len(distinct_catalog),
        "distinct_works": len(distinct_works),
        "duplicate_slots": dup["duplicate_slot_share"],
        "affected_users": dup["affected_user_share"],
        "fill_rate": filled / (len(user_ids) * recommended.shape[1]),
        "catalog_isbns": distinct_catalog,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=10)
    parser.add_argument("--limit-users", type=int, default=None)
    parser.add_argument("--gallery", action="store_true", default=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    catalog = load()
    split = make_split(catalog.ratings)
    train = build_interactions(split.train, weights="binary")
    works = catalog.works
    catalog_isbns = set(catalog.books["ISBN"])
    print(split.describe())

    test = split.test
    if args.limit_users:
        rng = np.random.default_rng(split.seed)
        sample = rng.choice(test["User-ID"].to_numpy(), size=min(args.limit_users, len(test)), replace=False)
        test = test[test["User-ID"].isin(set(sample))]
        print(f"RUNTIME GUARDRAIL: {len(test):,} users (seeded sample)")
    user_ids = test["User-ID"].to_numpy()
    holdout = test["ISBN"].to_numpy()
    print(f"data ready in {time.perf_counter() - started:.0f}s\n", flush=True)

    rows: list[dict] = []
    fitted: list = []
    for name in args.models:
        t0 = time.perf_counter()
        model = fit_model(build_model(name), train, catalog, split.train)
        wrapped = WorkDeduped(model, works, oversample=args.oversample)
        fit_seconds = time.perf_counter() - t0

        def score(width: int, model=model) -> np.ndarray:
            chunks = [
                model.recommend(user_ids[start : start + 512], k=width) for start in range(0, len(user_ids), 512)
            ]
            return np.concatenate(chunks, axis=0)

        t0 = time.perf_counter()
        plain = score(args.k)
        deduped = wrapped.collapse(score(args.k * args.oversample), user_ids, args.k)
        score_seconds = time.perf_counter() - t0
        for label, recs in ((f"{name}", plain), (f"{name} + dedup", deduped)):
            rows.append(summarize(label, recs, holdout, train, works, catalog_isbns, user_ids))
        print(
            f"{name:<12} fit {fit_seconds:>5.0f}s  score {score_seconds:>5.0f}s   "
            f"dup {rows[-2]['duplicate_slots']:.1%} -> {rows[-1]['duplicate_slots']:.1%}   "
            f"HitRate {rows[-2]['hit_rate']:.4f} -> {rows[-1]['hit_rate']:.4f}",
            flush=True,
        )
        fitted.append((model, wrapped))

    print("\n" + "=" * 118)
    print("M11.5 · What work-level dedup costs and buys")
    print("=" * 118)
    header = (
        f"{'model':<24} {'HitRate@10':>10} {'Cov@10':>8} {'Novelty':>8} "
        f"{'dup slots':>10} {'dup users':>10} {'filled':>8}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['label']:<24} {row['hit_rate']:>10.4f} {row['coverage_catalog']:>7.3%} "
            f"{row['novelty']:>8.2f} {row['duplicate_slots']:>9.1%} {row['affected_users']:>9.1%} "
            f"{row['fill_rate']:>7.1%}"
        )

    print("\n" + "=" * 118)
    print("M11.6 · Coverage@10 on each counting basis (no dedup — this is the table's basis)")
    print("=" * 118)
    print(f"{'model':<24} {'catalogue ISBNs':>18} {'all rec ISBNs':>16} {'distinct works':>16}")
    for row in rows:
        if row["label"].endswith("dedup"):
            continue
        print(
            f"{row['label']:<24} {row['distinct_catalog']:>9,} {row['coverage_catalog']:>7.3%} "
            f"{row['distinct_all']:>8,} {row['coverage_all_isbns']:>6.2%} "
            f"{row['distinct_works']:>8,} {row['coverage_works']:>6.2%}"
        )

    by_label = {row["label"]: row for row in rows}
    if "item-item" in by_label and "tfidf" in by_label:
        a, b = by_label["item-item"]["catalog_isbns"], by_label["tfidf"]["catalog_isbns"]
        union = a | b
        union_works = set(works.of(sorted(union)).tolist())
        print(
            f"\ncomplementarity on the catalogue basis: item-item {len(a):,} + TF-IDF {len(b):,}, "
            f"overlap {len(a & b):,}, union {len(union):,} = {len(union) / CATALOG_SIZE:.1%} of the catalogue "
            f"({len(union_works):,} works = {len(union_works) / works.n_works:.1%})"
        )

    if args.gallery:
        print("\n" + "=" * 118)
        print("M11.5 · The 3-anchor gallery with dedup on")
        print("=" * 118)
        print(render_markdown(build_gallery([w for _, w in fitted], catalog, k=args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
