"""Choose the ALS hyperparameters on the same inner validation split as item-item.

Identical discipline to ``scripts/tune_item_item.py``: the grid is scored against a
leave-one-out holdout carved out of *train* (seed 43), never against the evaluation
holdout. Consistency matters as much as the discipline itself — if one model were tuned
on validation and another on test, the comparison table would be meaningless no matter
how carefully each row was measured.

    python scripts/tune_als.py
"""

from __future__ import annotations

import sys
import time

from recommender.data import build_interactions, load
from recommender.eval import evaluate
from recommender.models.als import ALSRecommender
from recommender.split import make_split

VALIDATION_SEED = 43
FACTOR_GRID = [64, 128]
ALPHA_GRID = [1.0, 5.0, 20.0]
REGULARIZATION_GRID = [0.05]


def main() -> int:
    catalog = load()
    split = make_split(catalog.ratings)
    inner = make_split(split.train, seed=VALIDATION_SEED)
    inner_train = build_interactions(inner.train, weights="binary")
    catalog_isbns = set(catalog.books["ISBN"])
    print(f"validation split (from train only): {inner.describe()}\n", flush=True)

    best = None
    print(f"{'factors':>8} {'alpha':>7} {'reg':>6} {'HitRate@10':>11} {'Coverage@10':>12} {'fit s':>7}", flush=True)
    for factors in FACTOR_GRID:
        for alpha in ALPHA_GRID:
            for regularization in REGULARIZATION_GRID:
                started = time.perf_counter()
                model = ALSRecommender(factors=factors, alpha=alpha, regularization=regularization)
                model.fit(inner_train, catalog, ratings=inner.train)
                fit_seconds = time.perf_counter() - started
                result = evaluate(model, inner, inner_train, catalog_isbns=catalog_isbns)
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
