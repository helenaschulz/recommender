"""Measure the question the demo actually asks: one book in, is the right book among its neighbours.

    python scripts/measure_anchor_hitrate.py                  # the M20 table + M22's two hybrid rows
    python scripts/measure_anchor_hitrate.py --anchor-seed 43 # the sensitivity check, not a result
    python scripts/measure_anchor_hitrate.py --rules cascade  # the M20 table exactly as published
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

**Milestone M22 — all three hybrid rules, not only the conservative one.** M19 measured three
combination rules on the *profile* query and the accuracy winner there is score fusion (L77),
not the backfill cascade (L76). Until M22 this script built exactly one hybrid, the cascade,
so the published anchor table held the conservative rule and not the winning one and the
answer to *"your best model is not in your product table — why?"* was "we did not measure it".
``--rules`` now takes any subset of :data:`recommender.models.hybrid.RULES`; each is the same
class over the same two already-fitted base models, so nothing is re-fitted and the eight
published rows must reproduce beside the new ones. That reproduction is an **assert** here
(:data:`PUBLISHED_ANCHOR`), not an eyeball.

**α is inherited, not re-tuned, and the script says so in the row's own notes.** L77 chose
α = 0.6 on the inner validation split of the *profile* query. Re-tuning it for the item query
needs a second inner draw plus a sweep; L77's own α curve is a broad plateau from 0.6 to 0.8
rather than a peak, and tuned fusion could not be told apart from parameter-free RRF there
(p = 0.867). So the row inherits the value and is labelled as inheriting it — an inherited
parameter that reads as a fitted one is the M20 failure mode repeating.
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
from recommender.eval import (
    anchor_comparison_table,
    evaluate_anchors,
    hit_vector,
    mcnemar,
    owned_items,
    wilson_interval,
)
from recommender.models import ALL_MODELS, build_model, fit_model
from recommender.models.als import ALSRecommender
from recommender.models.hybrid import CASCADE, FUSION, RRF, RULES, HybridRecommender
from recommender.split import Split, pick_anchors

CACHE = Path("artifacts/anchors")

#: M18's cached per-user hit vectors for the *profile* question. Present after
#: `scripts/measure_significance.py`. They are the other half of every paired test here.
HIT_CACHE = Path("artifacts/significance/hits_work_k10.npz")

#: M19's cached per-user top-k *lists* for the three hybrid rules, written by
#: `scripts/measure_hybrid.py --save`. Hits are re-derived from them with `hit_vector`, the
#: same function that scores every column here, so the retention figures for the hybrid rows
#: come from one definition of a hit rather than from a second scoring path.
HYBRID_HIT_CACHE = Path("artifacts/hybrid/rules_work_k10.npz")

#: display name here -> (key in HYBRID_HIT_CACHE, its published profile HitRate@10).
HYBRID_PROFILE = {
    "hybrid (cascade)": ("cascade (floor 0)", 0.0650),  # L76
    "hybrid (fusion)": ("fusion (alpha 0.6)", 0.0690),  # L77
    "hybrid (rrf)": ("rrf (k=60)", 0.0687),  # L78
}

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

#: The published *anchor* rows of L80, seed 42 — the eight rows M20 put in the ledger. This
#: is the assertion M22 owes its own new rows: adding a hybrid rule must not move a single
#: existing cell, because nothing about the existing rows is re-fitted or re-scored. Checked
#: only on the pinned draw and only on an uncapped run; any other run is not this table.
PUBLISHED_ANCHOR = {
    "hybrid (cascade)": 0.0317,
    "als": 0.0308,
    "item-item": 0.0297,
    "content-tfidf": 0.0258,
    "item-item (explicit-only)": 0.0208,
    "popularity": 0.0155,
    "als (no support floor)": 0.0130,
    "content-embeddings": 0.0126,
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


def hybrid_profile_hits(holdout: np.ndarray) -> dict[str, np.ndarray]:
    """The three hybrid rules' *profile* hit vectors, re-derived from M19's saved lists.

    Returns only the rules whose re-derived HitRate@10 reproduces its published cell, and
    prints the ones it drops. A retention figure computed against a number nobody re-checked
    is exactly the kind of row M20 exists to stop.
    """
    if not HYBRID_HIT_CACHE.exists():
        print(f"note: {HYBRID_HIT_CACHE} absent — the hybrid rows get no retention column this run")
        return {}
    stored = np.load(HYBRID_HIT_CACHE, allow_pickle=True)
    if not np.array_equal(stored["holdout"].astype(str), holdout.astype(str)):
        print(f"note: {HYBRID_HIT_CACHE} was written against a different holdout — not paired")
        return {}
    out: dict[str, np.ndarray] = {}
    for display, (key, published) in HYBRID_PROFILE.items():
        if key not in stored.files:
            continue
        lists = stored[key].astype(object)
        lists[lists == "None"] = None
        vector = hit_vector(lists, holdout)
        measured = round(float(vector.mean()), 4)
        if abs(measured - published) > 1e-9:
            print(f"REPRODUCTION FAILURE  {display}: cached profile HitRate@10 {measured}, published {published}")
            continue
        out[display] = vector
    if out:
        names = ", ".join(sorted(out))
        print(f"hybrid profile hits re-derived from {HYBRID_HIT_CACHE}, all reproduce their published cell: {names}")
    return out


def check_published_anchors(hits: dict[str, np.ndarray]) -> list[str]:
    """Every row M20 published must come back to its published digits. Returns the failures."""
    failures = []
    for name, published in PUBLISHED_ANCHOR.items():
        if name not in hits:
            continue
        measured = round(float(hits[name].mean()), 4)
        if abs(measured - published) > 1e-9:
            failures.append(f"{name}: measured AnchorHitRate@10 {measured}, published (L80) {published}")
    return failures


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


def strata_pairwise(hits: dict[str, np.ndarray], support: np.ndarray, opponent: str) -> pd.DataFrame:
    """Paired McNemar against one row, **within each anchor-support band**.

    L81 reports exactly this shape for ALS against item-item and had to compute it by hand.
    It is the table that decides whether a win on the aggregate means anything for the demo:
    the app refuses anchors below 50 readers (L65), so a rule whose advantage sits entirely
    in the thin bands wins the column and changes nothing on screen.
    """
    rows = []
    for name, vector in hits.items():
        if name == opponent:
            continue
        row: dict[str, object] = {"model": name, "against": opponent}
        for label, lo, hi in STRATA:
            mask = (support >= lo) if hi is None else ((support >= lo) & (support <= hi))
            test = mcnemar(vector[mask], hits[opponent][mask])
            p = f"{test['p']:.1e}" if test["p"] < 0.001 else f"{test['p']:.3f}"
            row[f"{label} (n={int(mask.sum()):,})"] = f"{test['a_only']}/{test['b_only']}, p={p}"
        rows.append(row)
    return pd.DataFrame(rows)


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
    parser.add_argument(
        "--rules",
        default=f"{CASCADE},{RRF},{FUSION}",
        help=f"comma-separated hybrid rules to score, any of {RULES}; empty string for none",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="score fusion's mixing weight — inherited from L77's profile tuning, NOT re-tuned here",
    )
    parser.add_argument("--reference", default="item-item", help="the row every other row is also tested against")
    args = parser.parse_args(argv)

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        print(f"FATAL  unknown rule(s) {unknown}; expected any of {RULES}")
        return 2

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
    if rules and {"item-item", "tfidf"} <= fitted.keys():
        # M19's three rules on the question the demo actually asks. Nothing is re-fitted: each
        # is the same wrapper class over the same two already-fitted base models, differing
        # only in a rule constant. Their candidate depth is fixed at 100 per base model by the
        # model itself, so on a reader with a very long shelf they have a shallower pool to
        # filter than the single models do — a property of the rules as M19 measured them,
        # recorded here rather than tuned away for this table.
        why = {
            CASCADE: "M19's recommended rule (L76), unfitted wrapper over the two fitted base models",
            RRF: "M19's parameter-free rule (L78): sum of 1/(60+rank), nothing tuned, nothing inherited",
            FUSION: (
                f"M19's accuracy winner on the profile query (L77). alpha={args.alpha} is INHERITED from "
                "the profile tuning on the inner validation split (seed 43) and was NOT re-tuned for the "
                "item query"
            ),
        }
        for rule in rules:
            extra.append(
                (
                    HybridRecommender(
                        fitted["item-item"],
                        fitted["tfidf"],
                        rule=rule,
                        alpha=args.alpha,
                        name=f"hybrid ({rule})",
                    ),
                    why[rule],
                )
            )

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
        # The ALS ablation has no profile row at all; each hybrid rule has one in M19's cache
        # rather than in M18's, and is looked up under its own display name.
        origin[result.model] = result.model if result.model in HYBRID_PROFILE else None
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

    # M22's own guard: the rows M20 published must be unmoved by the rows M22 adds. Reported
    # here where the numbers are, enforced by the exit code at the bottom so the rest of the
    # evidence still reaches the log rather than being lost to an early return.
    pinned_draw = cap == len(holdout) and args.anchor_seed == 42
    reproduction_failures = check_published_anchors(hits) if pinned_draw else []
    if pinned_draw:
        checked = sorted(set(PUBLISHED_ANCHOR) & set(hits))
        if reproduction_failures:
            for line in reproduction_failures:
                print(f"\nREPRODUCTION FAILURE  {line}")
        else:
            print(f"\nall {len(checked)} of L80's published anchor rows present in this run reproduce to the digit")

    print("\n=== By the anchor's train support (the same vectors, grouped) ===\n")
    print(strata_table(hits, support).to_markdown(index=False))

    paired: dict[str, np.ndarray] = {}
    if cap == len(holdout):
        paired.update(profile_hits(len(holdout)) or {})
        paired.update(hybrid_profile_hits(holdout))
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
            profile = float(paired[base].mean())
            anchor = float(vector.mean())
            comparisons.append(
                {
                    "model": name,
                    "profile query": round(profile, 4),
                    "anchor query": round(anchor, 4),
                    # L83's column, computed here rather than by hand afterwards: a new row in
                    # that table without its retention figure does not fit the table it joins.
                    "retained": f"{100 * anchor / profile:.1f}%" if profile else "n/a",
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
    print(pd.DataFrame(pairwise(hits, leader)).to_markdown(index=False))

    # Decision 4 of M22: the three rows that decide whether anything in the story changes.
    # item-item is the engine the demo could run, the cascade is the rule these would replace,
    # ALS is the row six documents credited with "the best neighbourhoods" until M20.
    for reference in dict.fromkeys([args.reference, "hybrid (cascade)", "als"]):
        if reference not in hits or reference == leader:
            continue
        print(f"\n=== Every model against {reference} ===\n")
        print(pd.DataFrame(pairwise(hits, reference)).to_markdown(index=False))
        print(f"\n--- the same comparison inside each anchor-support band (wins/losses against {reference}) ---\n")
        print(strata_pairwise(hits, support, reference).to_markdown(index=False))

    CACHE.mkdir(parents=True, exist_ok=True)
    # A capped run writes under its own name. Found by running one: the smoke test overwrote
    # the pinned seed-42 vectors with 400 rows, which is a ledger artefact silently replaced
    # by something that is explicitly "never a ledger row" three lines of help text above.
    tag = f"_capped{cap}" if cap < len(holdout) else ""
    vectors_file = CACHE / f"anchor_hits_work_k{args.k}_seed{args.anchor_seed}{tag}.npz"
    np.savez_compressed(vectors_file, **{name: v for name, v in hits.items()})
    draw_file = CACHE / f"anchors_work_seed{args.anchor_seed}{tag}.npz"
    np.savez_compressed(draw_file, anchors=anchor_ids.astype(str), users=user_ids, support=support)
    summary = {
        "k": args.k,
        "anchor_seed": args.anchor_seed,
        "n_users": int(len(holdout)),
        "capped_to": int(cap),
        "hybrid_rules": rules,
        "fusion_alpha_inherited_from_L77": args.alpha,
        "anchor_rule": anchors.describe(),
        "anchor_hit_rates": {r.model: round(r.anchor_hit_rate, 4) for r in results},
        "answered": {r.model: round(r.answered, 4) for r in results},
        "coverage": {r.model: round(r.coverage, 5) for r in results},
    }
    (CACHE / f"anchor_work_k{args.k}_seed{args.anchor_seed}{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nvectors cached to {vectors_file}, the draw itself to {draw_file}")
    print(
        "\nRead the fourth table as one sentence: for the same model and the same readers, "
        "does asking with one book do better or worse than asking with a whole history? A "
        "model whose two columns diverge is the reason this project reports both."
    )
    if reproduction_failures:
        print(
            f"\nFATAL  {len(reproduction_failures)} of L80's published rows did not reproduce — "
            "the new rows in this run are not comparable against the published table and "
            "nothing here is a ledger row."
        )
        return 1
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
