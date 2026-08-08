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

from recommender.benchmark import build_bench, ceilings
from recommender.eval import comparison_table, evaluate
from recommender.gallery import build_gallery, render_markdown
from recommender.models import ALL_MODELS, build_model, fit_model
from recommender.serving import WorkDeduped


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

    started = time.perf_counter()
    bench = build_bench(work_level=args.work_level)
    catalog, split, train = bench.catalog, bench.split, bench.train
    print(bench.describe())
    # Printed with every run, never carried over from another one: a HitRate read against
    # the ceiling of a different item universe is a wrong number, not a rounded one.
    print(ceilings(bench).describe())
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
        model = build_model(name, work_level=args.work_level)
        t0 = time.perf_counter()
        # One dispatch, shared with the notebook: see recommender.models.fit_model.
        fit_model(model, train, catalog, split.train)
        if args.dedup:
            model = WorkDeduped(model, bench.catalog.works)
        fit_seconds = time.perf_counter() - t0
        result = evaluate(
            model,
            split,
            train,
            catalog_isbns=bench.catalog_ids,
            catalog_size=bench.catalog_size,
            k=args.k,
            users=users,
            notes=f"evaluated on a seeded sample of {len(users):,} users" if users is not None else "",
        )
        print(f"{result}   [fit {fit_seconds:.0f}s]  params: {result.params}", flush=True)
        results.append(result)
        fitted.append(model)

    print("\n" + comparison_table(results).to_markdown(index=False))

    if args.gallery:
        print("\n" + render_markdown(build_gallery(fitted, catalog, anchors=bench.anchors, k=args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
