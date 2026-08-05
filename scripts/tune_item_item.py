"""Choose the item-item hyperparameters on a validation split carved out of train.

The point is what this script does *not* do: it never looks at the evaluation holdout.
Sweeping shrinkage on the test split and reporting the best number would be fitting to
the test set — the offline equivalent of marking your own homework, and the first thing
a careful reviewer would probe.

So the same leave-one-out procedure is applied a second time, to the training data only
(seed 43, one level deeper). Shrinkage and neighbourhood size are chosen on that inner
holdout; the chosen values are then run once against the real test split in
``scripts/run_model.py``.

    python scripts/tune_item_item.py
"""

from __future__ import annotations

import sys
import time

from recommender.data import build_interactions, load
from recommender.eval import evaluate
from recommender.models.item_item import ItemItemRecommender
from recommender.split import make_split

VALIDATION_SEED = 43
SHRINKAGE_GRID = [0.0, 10.0, 20.0, 50.0, 100.0]
NEIGHBOUR_GRID = [50, 200, 500]


def main() -> int:
    catalog = load()
    split = make_split(catalog.ratings)
    # One level deeper: hold out from train, never from split.test.
    inner = make_split(split.train, seed=VALIDATION_SEED)
    inner_train = build_interactions(inner.train, weights="binary")
    catalog_isbns = set(catalog.books["ISBN"])
    print(f"validation split (from train only): {inner.describe()}\n", flush=True)

    best = None
    print(f"{'shrinkage':>10} {'neighbours':>11} {'HitRate@10':>11} {'Coverage@10':>12} {'fit s':>7}")
    for shrinkage in SHRINKAGE_GRID:
        for neighbours in NEIGHBOUR_GRID:
            started = time.perf_counter()
            model = ItemItemRecommender(shrinkage=shrinkage, top_k_neighbours=neighbours)
            model.fit(inner_train, catalog)
            fit_seconds = time.perf_counter() - started
            result = evaluate(model, inner, inner_train, catalog_isbns=catalog_isbns)
            print(
                f"{shrinkage:>10.0f} {neighbours:>11} {result.hit_rate_at_10:>11.4f} "
                f"{result.coverage_at_10:>11.3%} {fit_seconds:>7.0f}",
                flush=True,
            )
            if best is None or result.hit_rate_at_10 > best[0]:
                best = (result.hit_rate_at_10, shrinkage, neighbours)

    print(f"\nchosen on validation: shrinkage={best[1]:.0f}, top_k_neighbours={best[2]} (HitRate@10={best[0]:.4f})")
    print("Run these against the real test split with scripts/run_model.py item-item.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
