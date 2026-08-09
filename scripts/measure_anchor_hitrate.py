"""Measure the question the demo actually asks: one book in, is the right book among its neighbours.

    python scripts/measure_anchor_hitrate.py                  # the M20 table
    python scripts/measure_anchor_hitrate.py --anchor-seed 43 # the sensitivity check, not a result
    python scripts/measure_anchor_hitrate.py --limit-users 500  # smoke test, never a ledger row

**Why this exists.** Every published accuracy row in this ledger scores a *user* query:
given this reader's history, rank their held-out book. The demo asks an *item* query: given
this one book, what is like it. The project has said those are two different questions since
M13 — and then answered the second one with three anchors read by eye (L34, 2026-08-04,
*before* the work-level re-base). After the re-base, item-item and ALS return the identical
*Harry Potter* neighbourhood (`docs/model_selection.md` §6), so no published number separates
them on the surface the app actually serves, while the sidebar and four documents still claim
one of them has "the best" neighbourhoods. This script replaces that claim with a column.

**The metric.** For each of the eligible users of the pinned work-level split, draw one
**anchor** from that reader's train rows (:func:`recommender.split.pick_anchors`), ask the
model for the anchor's top-10 neighbours with the reader's own train items filtered out, and
score a hit when the held-out book is among them. Same readers, same held-out books, same row
order as HitRate@10, and the hit is :func:`recommender.eval.hit_vector` unchanged — so the
two columns pair under McNemar and "the metric and the product surface disagree" becomes a
measured quantity rather than a sentence.

**What it does not settle.** A model may honestly win one column and lose the other; that is
the point of measuring both, not a defect. And the anchor rule is a **free parameter** — the
draw, its seed, its pool and its fallback count are reported with the number, and a second
seed is run as a labelled sensitivity check.

**Two caveats that are properties of the models, not of the metric, and both are printed.**
ALS applies a 20-interaction support floor to ``similar_items`` and to nothing else
(`als.py`, ledger L34), so its item-to-item candidate universe is not its HitRate@10 one; the
ablation without that floor is measured as its own row. And popularity's ``similar_items`` is
degenerate by construction — the global top-10 for every anchor — so its row is the control,
not a competitor.
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
from recommender.eval import anchor_comparison_table, evaluate_anchors, mcnemar, owned_items, wilson_interval
from recommender.models import ALL_MODELS, build_model, fit_model
from recommender.models.als import ALSRecommender
from recommender.models.hybrid import CASCADE, HybridRecommender
from recommender.split import Split, pick_anchors

CACHE = Path("artifacts/anchors")

#: M18's cached per-user hit vectors for the *profile* question. Present after
#: `scripts/measure_significance.py`. They are the other half of every paired test here.
HIT_CACHE = Path("artifacts/significance/hits_work_k10.npz")

#: The published work-level HitRate@10 of each model (L52-L57). Hard-coded on purpose: this
#: is the assertion, not an input. Only HitRate, because cached vectors can re-derive that
#: and nothing else — `scripts/measure_significance.py` is where all three are checked.
PUBLISHED = {
    "popularity": 0.0155,
    "item-item": 0.0644,
    "item-item-explicit": 0.0486,
    "tfidf": 0.0405,
    "als": 0.0545,
    "embeddings": 0.0141,
}

#: Anchor train-support bands. L73's boundaries with its leftmost bin dropped: an anchor is
#: itself a train interaction, so support 0 cannot occur and a 0 column would be a lie
#: shaped like a comparison. The 50+ band is also the app's askable-anchor floor (L65).
STRATA: tuple[tuple[str, int, int | None], ...] = (
    ("1-4", 1, 4),
    ("5-49", 5, 49),
    ("50+", 50, None),
)


def profile_hits(users: int) -> dict[str, np.ndarray] | None:
    """M18's per-user hit vectors, re-derived against the published cells before use.

    Returns None when the cache is absent or has drifted, so the run continues and reports
    the anchor column alone rather than pairing it against numbers nobody re-checked.
    """
    if not HIT_CACHE.exists():
        print(f"note: {HIT_CACHE} absent — no paired test against the profile column this run")
        return None
    stored = np.load(HIT_CACHE, allow_pickle=False)
    hits = {name: stored[name].astype(bool) for name in ALL_MODELS if name in stored.files}
    if any(len(v) != users for v in hits.values()):
        print(f"note: {HIT_CACHE} holds a different user count — not paired")
        return None
    drift = [
        f"{name}: cached HitRate@10 {round(float(v.mean()), 4)}, published {PUBLISHED[name]}"
        for name, v in hits.items()
        if abs(round(float(v.mean()), 4) - PUBLISHED[name]) > 1e-9
    ]
    if drift:
        for line in drift:
            print(f"REPRODUCTION FAILURE  {line}")
        return None
    print(f"profile hit vectors read from {HIT_CACHE} ({users:,} users), all six reproduce their published cell")
    return hits


def support_of(anchors: np.ndarray, train) -> np.ndarray:
    """Train interaction count of each anchor, 0 for anchors outside the train matrix."""
    popularity = train.item_popularity
    index = train.item_index
    return np.array([popularity[index[a]] if a in index else 0 for a in anchors], dtype=np.int64)


def pairwise(hits: dict[str, np.ndarray], opponent: str) -> list[dict[str, object]]:
    """Paired McNemar of every row against one named row."""
    if opponent not in hits:
        return []
    rows = []
    for name, vector in hits.items():
        if name == opponent:
            continue
        test = mcnemar(vector, hits[opponent])
        rows.append(
            {
                "model": name,
                "against": opponent,
                "wins": test["a_only"],
                "losses": test["b_only"],
                "\u0394": f"{test['diff']:+.4f}",
                "95% CI on \u0394": f"[{test['diff_lo']:+.4f}, {test['diff_hi']:+.4f}]",
                "McNemar p": f"{test['p']:.2e}" if test["p"] < 0.001 else f"{test['p']:.3f}",
                "verdict": "distinguishable" if test["p"] < 0.05 else "NOT distinguishable",
            }
        )
    return rows


def strata_table(hits: dict[str, np.ndarray], support: np.ndarray) -> pd.DataFrame:
    """The same vectors grouped by anchor support — one definition of a hit, grouped."""
    rows = []
    for name, vector in hits.items():
        row: dict[str, object] = {"model": name}
        for label, lo, hi in STRATA:
            mask = (support >= lo) if hi is None else ((support >= lo) & (support <= hi))
            rate = round(float(vector[mask].mean()), 4) if mask.any() else float("nan")
            row[f"{label} (n={int(mask.sum()):,})"] = rate
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--anchor-seed", type=int, default=42, help="the draw's seed; 42 is the pinned run")
    parser.add_argument("--limit-users", type=int, default=0, help="smoke test only — a capped run is not a row")
    parser.add_argument("--no-als-ablation", action="store_true", help="skip the no-support-floor ALS row")
    parser.add_argument("--no-hybrid", action="store_true", help="skip M19's cascade row")
    parser.add_argument("--reference", default="item-item", help="the row every other row is also tested against")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    bench = build_bench(work_level=True)
    anchors = pick_anchors(bench.split, seed=args.anchor_seed)
    print(bench.describe())
    print(anchors.describe())
    print(f"data ready in {time.perf_counter() - started:.0f}s\n", flush=True)

    anchor_ids = anchors.item_ids
    user_ids = anchors.user_ids
    if not np.array_equal(user_ids, bench.split.test["User-ID"].to_numpy()):
        print("FATAL  anchors are not aligned with split.test — every paired test below would be noise")
        return 1
    holdout = bench.split.test["ISBN"].to_numpy()
    if (anchor_ids == holdout).any():
        print("FATAL  an anchor equals its own held-out item — the split leaked into the query")
        return 1

    cap = args.limit_users or len(anchor_ids)
    if cap < len(anchor_ids):
        anchor_ids, user_ids = anchor_ids[:cap], user_ids[:cap]
    owned = owned_items(bench.train, user_ids)
    notes = f"CAPPED at {cap:,} users — smoke test, not a ledger row" if cap < len(holdout) else ""

    split = bench.split if cap == len(holdout) else _head(bench.split, cap)
    support = support_of(anchor_ids, bench.train)

    fitted: dict[str, object] = {}
    #: display name -> registry key, for the rows that have a published profile column.
    origin: dict[str, str | None] = {}
    results = []
    for name in ALL_MODELS:
        t0 = time.perf_counter()
        try:
            model = fit_model(build_model(name, work_level=True), bench.train, bench.catalog, bench.split.train)
        except SystemExit as exc:  # a missing optional dependency costs one column, not the sweep
            print(f"{name:<24} SKIPPED: {exc}", flush=True)
            continue
        fitted[name] = model
        result = evaluate_anchors(
            model,
            split,
            bench.train,
            anchor_ids,
            catalog_isbns=bench.catalog_ids,
            catalog_size=bench.catalog_size,
            k=args.k,
            owned=owned,
            notes=notes,
        )
        results.append(result)
        origin[result.model] = name
        print(f"{result}  [fit+score {time.perf_counter() - t0:.0f}s]", flush=True)

    extra: list[tuple[object, str]] = []
    if not args.no_als_ablation and "als" in fitted:
        ablation = ALSRecommender(similar_min_support=0)
        ablation.name = "als (no support floor)"
        extra.append((ablation, "ablation: the L34 floor removed, to price what the floor buys"))
    if not args.no_hybrid and {"item-item", "tfidf"} <= fitted.keys():
        # M19's recommended rule (L76), on the question the demo actually asks. Not re-fitted:
        # the cascade is a rule over two already-fitted models. Its candidate depth is fixed
        # at 100 per base model by the model itself, so on a reader with a very long shelf it
        # has a shallower pool to filter than the single models do — a property of the rule as
        # M19 measured it, recorded here rather than tuned away for this table.
        cascade = HybridRecommender(fitted["item-item"], fitted["tfidf"], rule=CASCADE, name="hybrid (cascade)")
        extra.append((cascade, "M19's recommended rule (L76), unfitted wrapper over the two fitted base models"))

    for model, note in extra:
        t0 = time.perf_counter()
        if isinstance(model, ALSRecommender):
            model = fit_model(model, bench.train, bench.catalog, bench.split.train)
        result = evaluate_anchors(
            model,
            split,
            bench.train,
            anchor_ids,
            catalog_isbns=bench.catalog_ids,
            catalog_size=bench.catalog_size,
            k=args.k,
            owned=owned,
            notes=note,
        )
        results.append(result)
        origin[result.model] = None  # an ablation and a hybrid have no published profile row
        print(f"{result}  [fit+score {time.perf_counter() - t0:.0f}s]", flush=True)

    print("\n=== The item-to-item table: one book in, ten books out ===\n")
    print(anchor_comparison_table(results).drop(columns=["params"]).to_markdown(index=False))

    hits = {r.model: r.hits for r in results}
    rows = []
    for name, vector in sorted(hits.items(), key=lambda kv: -kv[1].mean()):
        lo, hi = wilson_interval(int(vector.sum()), len(vector))
        rows.append(
            {
                "model": name,
                f"AnchorHitRate@{args.k}": round(float(vector.mean()), 4),
                "hits": int(vector.sum()),
                "95% CI (Wilson)": f"[{lo:.4f}, {hi:.4f}]",
            }
        )
    print("\n=== Intervals ===\n")
    print(pd.DataFrame(rows).to_markdown(index=False))

    print("\n=== By the anchor's train support (the same vectors, grouped) ===\n")
    print(strata_table(hits, support).to_markdown(index=False))

    paired = profile_hits(len(holdout)) if cap == len(holdout) else None
    if paired:
        comparisons = []
        for name, vector in hits.items():
            base = origin.get(name)
            # No string surgery on the display name: `tfidf` is published as "content-tfidf"
            # and an ad-hoc prefix match silently dropped it and embeddings from this table
            # on the first run. A row that vanishes is worse than a row that errors.
            if base is None or base not in paired:
                continue
            test = mcnemar(vector, paired[base])
            comparisons.append(
                {
                    "model": name,
                    "anchor query": round(float(vector.mean()), 4),
                    "profile query": round(float(paired[base].mean()), 4),
                    "anchor wins": test["a_only"],
                    "profile wins": test["b_only"],
                    "Δ": f"{test['diff']:+.4f}",
                    "95% CI on Δ": f"[{test['diff_lo']:+.4f}, {test['diff_hi']:+.4f}]",
                    "McNemar p": f"{test['p']:.2e}" if test["p"] < 0.001 else f"{test['p']:.3f}",
                }
            )
        print("\n=== The same model, the two questions (Δ is anchor minus profile) ===\n")
        print(pd.DataFrame(comparisons).to_markdown(index=False))

        print("\n=== Every model against the anchor-column leader ===\n")
        leader = max(hits, key=lambda n: hits[n].mean())
        against = []
        for name, vector in hits.items():
            if name == leader:
                continue
            test = mcnemar(vector, hits[leader])
            against.append(
                {
                    "model": name,
                    "against": leader,
                    "wins": test["a_only"],
                    "losses": test["b_only"],
                    "Δ": f"{test['diff']:+.4f}",
                    "McNemar p": f"{test['p']:.2e}" if test["p"] < 0.001 else f"{test['p']:.3f}",
                    "verdict": "distinguishable" if test["p"] < 0.05 else "NOT distinguishable",
                }
            )
        print(pd.DataFrame(against).to_markdown(index=False))

        print(f"\n=== Every model against {args.reference}, the model that wins the profile table ===\n")
        print(pd.DataFrame(pairwise(hits, args.reference)).to_markdown(index=False))

    CACHE.mkdir(parents=True, exist_ok=True)
    vectors_file = CACHE / f"anchor_hits_work_k{args.k}_seed{args.anchor_seed}.npz"
    np.savez_compressed(vectors_file, **{name: v for name, v in hits.items()})
    draw_file = CACHE / f"anchors_work_seed{args.anchor_seed}.npz"
    np.savez_compressed(draw_file, anchors=anchor_ids.astype(str), users=user_ids, support=support)
    summary = {
        "k": args.k,
        "anchor_seed": args.anchor_seed,
        "n_users": int(len(holdout)),
        "capped_to": int(cap),
        "anchor_rule": anchors.describe(),
        "anchor_hit_rates": {r.model: round(r.anchor_hit_rate, 4) for r in results},
        "answered": {r.model: round(r.answered, 4) for r in results},
        "coverage": {r.model: round(r.coverage, 5) for r in results},
    }
    (CACHE / f"anchor_work_k{args.k}_seed{args.anchor_seed}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nvectors cached to {vectors_file}, the draw itself to {draw_file}")
    print(
        "\nRead the fourth table as one sentence: for the same model and the same readers, "
        "does asking with one book do better or worse than asking with a whole history? A "
        "model whose two columns diverge is the reason this project reports both."
    )
    return 0


def _head(split, n: int):
    """A split truncated to its first *n* test rows, for smoke runs only."""
    return Split(
        train=split.train,
        test=split.test.head(n).reset_index(drop=True),
        seed=split.seed,
        min_explicit=split.min_explicit,
        relevance_threshold=split.relevance_threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
