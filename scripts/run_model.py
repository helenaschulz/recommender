"""Run one or more models on the pinned split and print ledger-ready numbers.

    python scripts/run_model.py popularity item-item
    python scripts/run_model.py --all --gallery
    python scripts/run_model.py item-item --work-level      # M11.4, the edition experiment
    python scripts/run_model.py tfidf --dedup --gallery     # M11.5, dedup at serving time

Thin wrapper: it wires data -> split -> model -> eval and prints. No scoring logic lives
here, so nothing measured through this script can differ from what the notebook measures
through the same package.

``--work-level`` and ``--dedup`` are two different answers to the same edition problem and
must not be confused. ``--work-level`` re-keys the *interaction data* to works before the
split, so the model trains on merged co-occurrence counts; it is an experiment, and its
numbers are not cell-comparable with the ISBN-level table (different item universe, and
the split's eligibility shifts with it). ``--dedup`` leaves every model and every training
number untouched and only collapses the *output*.
"""

from __future__ import annotations

import argparse
import sys
import time

from recommender.data import CATALOG_SIZE, build_interactions, load, to_work_level
from recommender.eval import comparison_table, evaluate
from recommender.gallery import build_gallery, render_markdown
from recommender.models import ALL_MODELS, build_model, fit_model
from recommender.serving import WorkDeduped
from recommender.split import make_split

#: These read `Books.csv` through the ISBN, so they cannot be run on work-keyed items
#: without a work-level catalogue — out of scope for the M11.4 experiment, which asks
#: only whether merging editions lifts the *collaborative* signal.
CATALOGUE_KEYED = {"tfidf", "embeddings"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help=f"run {', '.join(ALL_MODELS)}")
    parser.add_argument("--gallery", action="store_true", help="also print the 3-anchor gallery")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit-users", type=int, default=None, help="runtime guardrail: evaluate on a seeded sample")
    parser.add_argument("--work-level", action="store_true", help="M11.4: re-key interactions to works pre-split")
    parser.add_argument("--dedup", action="store_true", help="M11.5: collapse output to one ISBN per work at serving")
    args = parser.parse_args(argv)

    names = ALL_MODELS if args.all else args.models
    if not names:
        parser.error("name at least one model, or pass --all")
    if args.work_level and args.dedup:
        parser.error("--work-level already has one row per work; --dedup would be a no-op on top of it")
    if args.work_level and (blocked := sorted(set(names) & CATALOGUE_KEYED)):
        parser.error(f"--work-level cannot run {', '.join(blocked)}: they vectorize Books.csv per ISBN")
    if args.work_level and args.gallery:
        parser.error("--gallery anchors are ISBNs; use --dedup for the work-level demo surface")

    started = time.perf_counter()
    catalog = load()
    if args.work_level:
        works = catalog.works
        ratings = to_work_level(catalog.ratings, works)
        catalog_isbns = set(works.work_of_isbn)
        catalog_size = works.n_works
        print(
            f"WORK LEVEL: {len(catalog.books):,} ISBNs -> {catalog_size:,} works; "
            f"{len(catalog.ratings):,} interactions -> {len(ratings):,} rows "
            f"({len(catalog.ratings) - len(ratings):,} collapsed as same user + same work)"
        )
    else:
        ratings = catalog.ratings
        catalog_isbns = set(catalog.books["ISBN"])
        catalog_size = CATALOG_SIZE

    split = make_split(ratings)
    train = build_interactions(split.train, weights="binary")
    print(split.describe())
    print(f"data ready in {time.perf_counter() - started:.0f}s\n", flush=True)

    users = None
    if args.limit_users:
        import numpy as np

        rng = np.random.default_rng(split.seed)
        eligible = split.test["User-ID"].to_numpy()
        users = rng.choice(eligible, size=min(args.limit_users, len(eligible)), replace=False)
        print(f"RUNTIME GUARDRAIL: evaluating on a seeded sample of {len(users):,} users\n")

    results = []
    fitted = []
    for name in names:
        model = build_model(name)
        t0 = time.perf_counter()
        # One dispatch, shared with the notebook: see recommender.models.fit_model.
        fit_model(model, train, catalog, split.train)
        if args.dedup:
            model = WorkDeduped(model, catalog.works)
        fit_seconds = time.perf_counter() - t0
        result = evaluate(
            model,
            split,
            train,
            catalog_isbns=catalog_isbns,
            catalog_size=catalog_size,
            k=args.k,
            users=users,
            notes=f"evaluated on a seeded sample of {len(users):,} users" if users is not None else "",
        )
        print(f"{result}   [fit {fit_seconds:.0f}s]  params: {result.params}", flush=True)
        results.append(result)
        fitted.append(model)

    print("\n" + comparison_table(results).to_markdown(index=False))

    if args.gallery:
        print("\n" + render_markdown(build_gallery(fitted, catalog, k=args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
