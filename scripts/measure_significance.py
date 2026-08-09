"""Put an interval around every published HitRate, and a paired test under every comparison.

    python scripts/measure_significance.py                    # work level: the published table
    python scripts/measure_significance.py --isbn-level       # the journey table, same treatment

**Why this exists.** The ledger's whole claim is honesty, and until this ran it carried
sixty-nine lines of point estimates with no uncertainty anywhere. One published comparison
does not survive that omission: popularity 0.0155 against content embeddings 0.0141 is
nineteen users out of 13,580. The comparison table already carries a *derivation* of that
(unpaired, `sqrt(p(1-p)/n)`, z ≈ 1.0) and the two cells were reworded on it. A derivation
is an argument; this script is the measurement, and the two are allowed to disagree.

**Why the test is paired.** Every model is scored on the *same* users with the *same*
held-out book, so the unpaired standard error is the wrong instrument twice over: it throws
away the pairing, and on this data the pairing is strong — the users a model can hit at all
are largely the users with enough interaction evidence to be reachable (L27), so the models
agree far more than two independent samples would. **McNemar** looks only at the users where
the two models disagree: `b` users that A hit and B missed, `c` the other way round. Under
the null "the two are equally likely to be the one that hits", `b` is Binomial(b + c, 0.5),
and the two-sided exact binomial p on that is the whole test. Concordant users carry no
information about which model is better and are correctly ignored.

**The two tests live in :mod:`recommender.eval`** (:func:`~recommender.eval.wilson_interval`
and :func:`~recommender.eval.mcnemar`) rather than in this script, because M19's hybrid rows
have to be tested the same way and a second copy of a statistical test is a second answer.

**Two intervals, and they answer different questions.** The **Wilson** 95% interval on each
HitRate says how precisely that single number is pinned by 13,580 users — it is what belongs
next to a cell in the table. The **paired difference** interval, `(b - c) / n` with standard
error `sqrt(b + c) / n`, says how precisely the *gap between two cells* is pinned, and it is
always the tighter of the two for correlated models. Reporting only the first would make
every comparison in the table look less certain than it is; reporting only the second would
leave the cells themselves unqualified.

**The reproduction check is the point, not a formality.** This script re-runs the six models
through the same :func:`recommender.eval.evaluate` the comparison table was produced by, so
**all three published metrics** — HitRate, Coverage and Novelty — must come back equal to the
digits the table prints. :data:`PUBLISHED` holds those values for both tables, and a mismatch
is a hard failure that exits non-zero: an interval around a number that has quietly moved is
worse than no interval at all. Per-user hit vectors are cached to ``artifacts/significance/``
so a second question about the same run costs no compute; a cached run can only re-derive
HitRate, and it says so rather than implying the other two were re-checked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.benchmark import build_bench
from recommender.eval import evaluate, hit_vector, mcnemar, wilson_interval
from recommender.models import ALL_MODELS, build_model, fit_model

CACHE = Path("artifacts/significance")

#: Every published cell of both comparison tables — HitRate@10, Coverage@10 as a
#: **percentage** and Novelty@10 — work level (the primary table) and ISBN level (the
#: journey table). Hard-coded on purpose: this is the assertion, not an input. All three
#: metrics rather than only the one this script tests, because "the numbers still reproduce"
#: is a claim about the table, and a run that re-fits six models has already paid for it.
PUBLISHED: dict[str, dict[str, tuple[float, float, float]]] = {
    "work": {
        "popularity": (0.0155, 0.027, 10.54),
        "item-item": (0.0644, 8.190, 14.17),
        "item-item-explicit": (0.0486, 10.036, 15.97),
        "tfidf": (0.0405, 16.806, 17.07),
        "als": (0.0545, 0.897, 12.29),
        "embeddings": (0.0141, 26.143, 18.34),
    },
    "isbn": {
        "popularity": (0.0145, 0.019, 10.81),
        "item-item": (0.0546, 9.064, 14.91),
        "item-item-explicit": (0.0379, 10.739, 16.59),
        "tfidf": (0.0228, 16.616, 17.63),
        "als": (0.0451, 0.835, 12.63),
        "embeddings": (0.0109, 23.911, 18.42),
    },
}


def check_row(name: str, result, published: dict[str, tuple[float, float, float]]) -> list[str]:
    """Compare one freshly measured row against its published cell, all three metrics.

    Returns a list of human-readable failures, empty when the row reproduces. Coverage is
    compared as a percentage to three decimals and Novelty to two, because that is the
    precision the tables publish — asserting more digits than were ever printed would fail
    on formatting rather than on drift.
    """
    want_hit, want_cov, want_nov = published[name]
    got = (
        round(result.hit_rate_at_10, 4),
        round(result.coverage_at_10 * 100, 3),
        round(result.novelty_at_10, 2),
    )
    labels = ("HitRate@10", "Coverage@10 %", "Novelty@10")
    return [
        f"{name}: {label} measured {measured}, published {expected}"
        for label, measured, expected in zip(labels, got, (want_hit, want_cov, want_nov), strict=True)
        if abs(measured - expected) > 1e-9
    ]


def hit_vectors(*, work_level: bool, k: int, verbose: bool) -> tuple[dict[str, np.ndarray], list[str]]:
    """Fit every model on the pinned split and return one boolean hit vector per model.

    The vectors come out of the same :class:`recommender.eval.EvalResult` the comparison
    table is built from — one code path, so the HitRates cannot drift apart from the
    published ones without the check catching it. All three published metrics are checked
    here, while the fitted models are still in hand; the returned list is the failures.
    """
    started = time.perf_counter()
    bench = build_bench(work_level=work_level)
    print(bench.describe())
    print(f"data ready in {time.perf_counter() - started:.0f}s\n", flush=True)

    holdout = bench.split.test["ISBN"].to_numpy()

    published = PUBLISHED["work" if work_level else "isbn"]
    failures: list[str] = []
    hits: dict[str, np.ndarray] = {}
    for name in ALL_MODELS:
        t0 = time.perf_counter()
        model = fit_model(build_model(name, work_level=work_level), bench.train, bench.catalog, bench.split.train)
        result = evaluate(
            model,
            bench.split,
            bench.train,
            catalog_isbns=bench.catalog_ids,
            catalog_size=bench.catalog_size,
            k=k,
        )
        vector = hit_vector(result.recommendations, holdout)
        hits[name] = vector
        assert abs(float(vector.mean()) - result.hit_rate_at_10) < 1e-12, "hit vector disagrees with the metric"
        row_failures = check_row(name, result, published) if k == 10 else []
        failures += row_failures
        if verbose:
            status = "reproduces" if not row_failures else "DRIFTED"
            print(
                f"{name:<20} HitRate@{k}={result.hit_rate_at_10:.4f}  "
                f"Coverage@{k}={result.coverage_at_10:.3%}  Novelty@{k}={result.novelty_at_10:.2f}  "
                f"[{time.perf_counter() - t0:.0f}s]  {status}",
                flush=True,
            )
    return hits, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isbn-level", action="store_true", help="the journey table instead of the published one")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--reference", default="item-item", help="the row every other row is tested against")
    parser.add_argument("--baseline", default="popularity")
    parser.add_argument("--no-cache", action="store_true", help="ignore any cached hit vectors and re-fit")
    args = parser.parse_args(argv)

    level = "isbn" if args.isbn_level else "work"
    cache_file = CACHE / f"hits_{level}_k{args.k}.npz"

    published = PUBLISHED[level]
    from_cache = cache_file.exists() and not args.no_cache
    if from_cache:
        stored = np.load(cache_file, allow_pickle=False)
        hits = {name: stored[name].astype(bool) for name in ALL_MODELS}
        n_users = len(next(iter(hits.values())))
        # A cached run checked all three metrics when it was written; from the cache only
        # HitRate can be re-derived, and it is re-derived rather than trusted.
        failures = [
            f"{name}: HitRate@10 measured {round(float(v.mean()), 4)}, published {published[name][0]}"
            for name, v in hits.items()
            if abs(round(float(v.mean()), 4) - published[name][0]) > 1e-9
        ]
        print(f"per-user hit vectors read from {cache_file} ({n_users:,} users)\n")
    else:
        hits, failures = hit_vectors(work_level=not args.isbn_level, k=args.k, verbose=True)
        n_users = len(next(iter(hits.values())))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, **hits)
        print(f"\nper-user hit vectors cached to {cache_file}\n")

    # --- The reproduction check. A moved number stops the milestone. -------------------
    if failures:
        for line in failures:
            print(f"REPRODUCTION FAILURE  {line}")
        print("\nA published number moved. Stop and escalate — do not write intervals around a moved number.")
        return 1
    scope = "HitRate@10 (cached vectors carry no coverage or novelty)" if from_cache else "every published cell"
    print(f"reproduction check: {scope} of the {level}-level table reproduces\n")

    # --- One interval per cell ---------------------------------------------------------
    rows = []
    for name, vector in sorted(hits.items(), key=lambda kv: -kv[1].mean()):
        n_hits = int(vector.sum())
        lo, hi = wilson_interval(n_hits, n_users)
        rows.append(
            {
                "model": name,
                f"HitRate@{args.k}": round(float(vector.mean()), 4),
                "hits": n_hits,
                "95% CI (Wilson)": f"[{lo:.4f}, {hi:.4f}]",
                "±": f"{(hi - lo) / 2:.4f}",
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))

    # --- One paired test per comparison ------------------------------------------------
    comparisons = []
    for opponent in (args.reference, args.baseline):
        for name, vector in hits.items():
            if name == opponent:
                continue
            test = mcnemar(vector, hits[opponent])
            comparisons.append(
                {
                    "model": name,
                    "against": opponent,
                    "wins": test["a_only"],
                    "losses": test["b_only"],
                    "discordant": test["discordant"],
                    "Δ HitRate": f"{test['diff']:+.4f}",
                    "95% CI on Δ": f"[{test['diff_lo']:+.4f}, {test['diff_hi']:+.4f}]",
                    "McNemar p": f"{test['p']:.2e}" if test["p"] < 0.001 else f"{test['p']:.3f}",
                    "verdict": "distinguishable" if test["p"] < 0.05 else "NOT distinguishable",
                }
            )
    print("\n" + pd.DataFrame(comparisons).to_markdown(index=False))

    summary = {
        "level": level,
        "k": args.k,
        "n_users": n_users,
        "hit_rates": {name: round(float(v.mean()), 4) for name, v in hits.items()},
        "wilson": {name: wilson_interval(int(v.sum()), n_users) for name, v in hits.items()},
        "comparisons": comparisons,
    }
    out = CACHE / f"significance_{level}_k{args.k}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {out}")
    print(
        "\nRead the second table as one sentence per row: of the users where exactly one of "
        "the two models found the held-out book, how lopsided was the split? A row whose "
        "interval on Δ crosses zero is a pair of cells this dataset cannot order."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
