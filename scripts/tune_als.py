"""Choose the ALS hyperparameters on the same inner validation split as item-item.

Identical discipline to ``scripts/tune_item_item.py``: the grid is scored against a
leave-one-out holdout carved out of *train* (seed 43), never against the evaluation
holdout. Consistency matters as much as the discipline itself — if one model were tuned
on validation and another on test, the comparison table would be meaningless no matter
how carefully each row was measured.

    python scripts/tune_als.py
"""

from __future__ import annotations

import argparse
import sys
import time

from recommender.benchmark import build_bench, inner_bench
from recommender.eval import evaluate
from recommender.models.als import ALSRecommender

VALIDATION_SEED = 43
FACTOR_GRID = [64, 128]
ALPHA_GRID = [1.0, 5.0, 20.0]
REGULARIZATION_GRID = [0.05]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-level", action="store_true", help="tune on the work-keyed matrix (M12.2)")
    args = parser.parse_args(argv)

    # The inner universe is carved out of train and, at work level, its canonical text
    # excludes both holdouts -- see recommender.benchmark.
    outer = build_bench(work_level=args.work_level)
    inner = inner_bench(outer, seed=VALIDATION_SEED)
    catalog, inner_train = inner.catalog, inner.train
    catalog_isbns = inner.catalog_ids
    print(f"{inner.item_level}-level validation split (from train only): {inner.split.describe()}\n", flush=True)
    inner = inner.split

    best = None
    print(f"{'factors':>8} {'alpha':>7} {'reg':>6} {'HitRate@10':>11} {'Coverage@10':>12} {'fit s':>7}", flush=True)
    for factors in FACTOR_GRID:
        for alpha in ALPHA_GRID:
            for regularization in REGULARIZATION_GRID:
                started = time.perf_counter()
                model = ALSRecommender(factors=factors, alpha=alpha, regularization=regularization)
                model.fit(inner_train, catalog, ratings=inner.train)
                fit_seconds = time.perf_counter() - started
                result = evaluate(
                    model, inner, inner_train, catalog_isbns=catalog_isbns, catalog_size=outer.catalog_size
                )
                print(
                    f"{factors:>8} {alpha:>7.0f} {regularization:>6.2f} {result.hit_rate_at_10:>11.4f} "
                    f"{result.coverage_at_10:>11.3%} {fit_seconds:>7.0f}",
                    flush=True,
                )
                if best is None or result.hit_rate_at_10 > best[0]:
                    best = (result.hit_rate_at_10, factors, alpha, regularization)

    print(f"\nchosen on validation: factors={best[1]}, alpha={best[2]:.0f}, regularization={best[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
