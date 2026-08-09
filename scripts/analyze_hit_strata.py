"""L27 and L28 at work level: the same two strata, on the table the project actually publishes.

    python scripts/analyze_hit_strata.py                 # reads the cached hit vectors
    python scripts/analyze_hit_strata.py --no-cache      # re-fits the six models first

**Why this is its own script.** L27 ("the baseline is narrow, not weak") and L28 ("item-item
degrades on long profiles") were measured on the ISBN-level table in M5/M6 and never
recomputed after M12 moved the primary comparison to works. `notebooks/02_models.ipynb` §2.1
*did* recompute the popularity half — it prints the four support strata — but a number that
lives only in a notebook output is measured and unrecorded, which is the state the ledger
exists to prevent. This script produces both strata for every model, so the line can be
written and the notebook and the ledger can stop disagreeing about what exists.

**Two ways of slicing the same 13,580 users, and they answer opposite questions.**

- *By the held-out work's train support* — how much interaction evidence the **target**
  carries. This is the stratum that explains why the baseline's aggregate is not a weak
  score but a narrow one, and it is the stratum the hybrid argument stands on: whatever the
  content layer reaches, it reaches in the two leftmost columns.
- *By the reader's train profile length* — how much evidence the **query** carries. The
  item-KNN weakness L28 recorded lives here: summing similarities over a long profile lets
  volume drown the signal.

**No re-fitting if it can be helped.** The strata are the per-user hit vectors from the
published run, grouped — so they are computed from the cache
``scripts/measure_significance.py`` writes, and every aggregate is asserted against the
published HitRate before any stratum is printed. A stratum table under a moved aggregate
would be four wrong numbers instead of one.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from recommender.benchmark import build_bench
from recommender.eval import evaluate, hit_vector
from recommender.models import ALL_MODELS, build_model, fit_model

CACHE = "artifacts/significance/hits_{level}_k{k}.npz"

#: Support of the held-out item in the train matrix. The boundaries are L27's, unchanged,
#: so the work-level line can be read against the ISBN-level one column by column.
SUPPORT_STRATA = [(0, 0, "0 (unreachable)"), (1, 4, "1-4"), (5, 49, "5-49"), (50, 10**9, "50+")]

#: Length of the reader's train profile. L28's boundaries, likewise unchanged.
PROFILE_STRATA = [(0, 9, "0-9"), (10, 24, "10-24"), (25, 74, "25-74"), (75, 10**9, "75+")]


def stratify(hits: np.ndarray, values: np.ndarray, strata: list[tuple[int, int, str]]) -> dict[str, float]:
    """HitRate within each stratum, plus the user count that carries it."""
    out: dict[str, float] = {}
    for lo, hi, label in strata:
        mask = (values >= lo) & (values <= hi)
        out[label] = float(hits[mask].mean()) if mask.any() else float("nan")
        out[f"n {label}"] = int(mask.sum())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isbn-level", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true", help="re-fit the six models instead of reading the cache")
    args = parser.parse_args(argv)

    level = "isbn" if args.isbn_level else "work"
    bench = build_bench(work_level=not args.isbn_level)
    print(bench.describe() + "\n")

    holdout = bench.split.test["ISBN"].to_numpy()
    users = bench.split.test["User-ID"].to_numpy()

    # The two things a user is stratified by, both taken on **train only**: a support count
    # that included the holdout would be the metric grading its own homework.
    support = np.array(
        [
            bench.train.item_popularity[bench.train.item_index[item]] if item in bench.train.item_index else 0
            for item in holdout
        ]
    )
    matrix = bench.train.matrix
    profile_length = np.array(
        [
            (lambda row: 0 if row is None else int(matrix.indptr[row + 1] - matrix.indptr[row]))(
                bench.train.user_index.get(int(user))
            )
            for user in users
        ]
    )

    path = CACHE.format(level=level, k=args.k)
    hits: dict[str, np.ndarray] = {}
    try:
        if args.no_cache:
            raise FileNotFoundError
        stored = np.load(path, allow_pickle=False)
        hits = {name: stored[name].astype(bool) for name in ALL_MODELS}
        print(f"per-user hit vectors read from {path}\n")
    except FileNotFoundError:
        print("no cache — re-fitting the six models (about 8 minutes)\n", flush=True)
        for name in ALL_MODELS:
            model = build_model(name, work_level=not args.isbn_level)
            fit_model(model, bench.train, bench.catalog, bench.split.train)
            result = evaluate(
                model,
                bench.split,
                bench.train,
                catalog_isbns=bench.catalog_ids,
                catalog_size=bench.catalog_size,
                k=args.k,
            )
            hits[name] = hit_vector(result.recommendations, holdout)
            print(f"{name:<20} HitRate@{args.k}={result.hit_rate_at_10:.4f}", flush=True)

    if len(next(iter(hits.values()))) != len(holdout):
        print("cached vectors do not line up with this split — re-run with --no-cache")
        return 1

    by_support, by_profile = [], []
    for name, vector in hits.items():
        aggregate = round(float(vector.mean()), 4)
        by_support.append({"model": name, f"HitRate@{args.k}": aggregate, **stratify(vector, support, SUPPORT_STRATA)})
        by_profile.append(
            {"model": name, f"HitRate@{args.k}": aggregate, **stratify(vector, profile_length, PROFILE_STRATA)}
        )

    counts = {label: int(((support >= lo) & (support <= hi)).sum()) for lo, hi, label in SUPPORT_STRATA}
    print(f"held-out work support, users per stratum: {counts}")
    print(
        pd.DataFrame(by_support)
        .drop(columns=[c for c in by_support[0] if c.startswith("n ")])
        .round(4)
        .to_markdown(index=False)
    )

    counts = {label: int(((profile_length >= lo) & (profile_length <= hi)).sum()) for lo, hi, label in PROFILE_STRATA}
    print(f"\nreader train-profile length, users per stratum: {counts}")
    print(
        pd.DataFrame(by_profile)
        .drop(columns=[c for c in by_profile[0] if c.startswith("n ")])
        .round(4)
        .to_markdown(index=False)
    )
    print(
        "\nThe left column of the first table is the hybrid argument in one number: it is the "
        "share of held-out works no collaborative model can rank at all, and what a content "
        "layer would have to score in to be worth its complexity."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
