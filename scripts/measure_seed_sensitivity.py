"""Every number in this ledger is conditional on one draw. This runs five and reports the spread.

    python scripts/measure_seed_sensitivity.py --drift-only   # the gate, no model fitted
    python scripts/measure_seed_sensitivity.py                # the sweep: 42, 44, 45, 46, 47

**What this covers that L74 does not.** The Wilson intervals and paired McNemar of M18 hold the
drawn holdout **fixed** and quantify sampling across *readers*. Seed 42 also decides **which**
of a reader's favourites is withdrawn, and the paragraph under that table names this as the
larger uncovered source and records that it was not done. This is that measurement.

**Why it is not academic.** L73 has item-item at 0.0000 / 0.0170 / 0.0571 / 0.1198 across the
four support bands — factor seven, and the draw decides which band a reader's book lands in.
The models are strong in *different* bands, so a draw that reaches deeper into the tail does
not move all six rows down together, it moves them **against each other**. The claim under
test is the **ordering**, not the level.

**The headline is the paired per-seed delta, not the spread of levels.** Five levels tell you
the number wobbles; five deltas tell you whether the ordering wobbles, which is the only thing
the project leans on. The three margins that could plausibly move, reported first: ALS against
item-item (0.0099 apart, the narrowest pair L74 still calls distinguishable), the M19 fusion
rules against item-item (+0.0046, barely above L74's stated resolution), and embeddings
against popularity (published as *not* distinguishable, p = 0.342 — a seed that makes it
distinguishable in either direction is worth knowing before a reviewer asks).

**Seed 43 is deliberately skipped.** It already names the *inner validation* split in L51 and
L77, and two different draws sharing a number in one ledger is how a reader ends up believing
the sweep tuned on its own test set.

**The channel this does not close, stated rather than hidden.** λ=10 / 50 neighbours and ALS
128 / α=1 were selected on validation carved from **seed 42's** train, so under seed 47 an item
now held out did take part in that sweep. It is real and it is small — both sweeps landed on
plateaus, and L51 (the work-level re-tune that re-selected the identical values) is the
evidence that the choice does not depend on the split it was tuned on. Re-tuning five times is
5x the sweep cost and out of budget for this milestone.

**The gate, and why it runs before anything is fitted.** ``work_level_catalog`` picks each
work's canonical title from its most-interacted edition *counted on train only*, so a different
holdout can flip that argmax. Two consequences, and the second is the dangerous one:

1. If the text drifts, the embedding vectors cached under ``artifacts/embeddings/`` were
   computed from *different* text than this seed's catalogue.
2. :func:`recommender.models.embeddings._fingerprint` hashes the model name, the array length
   and **every ``len(texts)//512``-th text** — roughly 512 samples of 235,824. It is therefore
   perfectly capable of **not noticing** a drift and serving the wrong cache silently. A cache
   miss costs a re-encode; a stale cache hit costs the correctness of the row.

So the drift is measured on the **full** array, with no model in the loop. It is not zero
(78 works of 235,824 on this data) and the fingerprint does not notice. The sweep therefore
reuses the cache *knowingly* and **bounds the error rather than hiding it**: the embeddings
row is exact for every work but those, and the count of drifted works that are actually
somebody's held-out book is printed per seed, because that is what caps how far the metric
could move. Re-encoding 235,824 works per seed is two hours a seed and out of budget; the fix
to ``_fingerprint`` is out of scope here for the reason M21's own escalation rule gives.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.benchmark import build_bench
from recommender.data import load
from recommender.eval import evaluate, hit_vector, mcnemar
from recommender.models import ALL_MODELS, build_model, fit_model
from recommender.models.content_tfidf import content_text
from recommender.models.embeddings import _fingerprint
from recommender.models.hybrid import CASCADE, FUSION, RRF, HybridRecommender

CACHE = Path("artifacts/seeds")

#: M18's cached per-user hit vectors for seed 42. The sweep re-derives seed 42 through the
#: same `evaluate` and checks bit-for-bit against these, because "four decimals agree" is a
#: weaker check than 13,580 booleans (the position `scripts/measure_hybrid.py` already takes).
HIT_CACHE = Path("artifacts/significance/hits_work_k10.npz")

#: The published work-level table (L52–L57), as (HitRate@10, Coverage@10 %, Novelty@10).
#: Hard-coded: this is the assertion, not an input. Seed 42 must reproduce all three.
PUBLISHED = {
    "popularity": (0.0155, 0.027, 10.54),
    "item-item": (0.0644, 8.190, 14.17),
    "item-item-explicit": (0.0486, 10.036, 15.97),
    "tfidf": (0.0405, 16.806, 17.07),
    "als": (0.0545, 0.897, 12.29),
    "embeddings": (0.0141, 26.143, 18.34),
}

#: 43 is missing on purpose — see the module docstring.
DEFAULT_SEEDS = (42, 44, 45, 46, 47)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

#: The comparisons worth reporting per seed. Everything else in the table is separated by
#: p < 1e-05 and will not care which book was withdrawn.
MARGINS = (
    ("als", "item-item"),
    ("embeddings", "popularity"),
    ("hybrid-rrf", "item-item"),
)


def drift_gate(seeds: tuple[int, ...], catalog) -> tuple[dict[int, int], dict[int, set[str]]]:
    """Canonical-text drift per seed against the first, measured with no model in the loop.

    Returns ``(drift_per_seed, drifted_ids_per_seed)``. Never trust the fingerprint here:
    it samples ~512 of 235,824 texts and can miss a drift entirely, which is exactly what it
    does on this data.
    """
    print("=== Gate: does the canonical text survive a seed change? ===\n", flush=True)
    reference_ids = reference_text = reference_fp = None
    drift: dict[int, int] = {}
    drifted_ids: dict[int, set[str]] = {}
    honest = True
    for seed in seeds:
        started = time.perf_counter()
        bench = build_bench(work_level=True, catalog=catalog, seed=seed)
        books = bench.catalog.books
        order = np.argsort(books["ISBN"].to_numpy())
        ids = books["ISBN"].to_numpy()[order]
        texts = content_text(books)[order]
        fingerprint = _fingerprint(EMBEDDING_MODEL, texts)

        if reference_text is None:
            reference_ids, reference_text, reference_fp = ids, texts, fingerprint
            drift[seed] = 0
            drifted_ids[seed] = set()
            print(f"seed {seed}: {len(ids):,} works, fingerprint {fingerprint} (reference) "
                  f"[{time.perf_counter() - started:.0f}s]", flush=True)
            continue

        if not np.array_equal(ids, reference_ids):
            raise SystemExit(f"seed {seed}: the work universe itself moved — the seeds are not comparable")
        moved = texts != reference_text
        changed = int(moved.sum())
        drift[seed] = changed
        drifted_ids[seed] = set(ids[moved].tolist())
        notices = fingerprint != reference_fp
        honest = honest and (changed == 0 or notices)
        verdict = "cache invalidated" if notices else "cache would be REUSED"
        print(f"seed {seed}: canonical-text drift {changed:,} of {len(ids):,} works "
              f"({changed / len(ids):.4%}), fingerprint {fingerprint} — {verdict} "
              f"[{time.perf_counter() - started:.0f}s]", flush=True)
        if changed:
            for i in np.flatnonzero(texts != reference_text)[:3]:
                print(f"    {reference_ids[i]}: {reference_text[i]!r} -> {texts[i]!r}")

    union = set().union(*drifted_ids.values())
    print()
    if not union:
        print("Gate: no drift on any seed. The embedding cache is exact for all of them.\n")
    else:
        print(f"Gate: {len(union):,} distinct works drift somewhere in the sweep "
              f"({len(union) / len(reference_ids):.4%} of the catalogue).")
        if not honest:
            print("  **The fingerprint does not notice.** `_fingerprint` hashes the model name, the\n"
                  "  array length and every len/512-th text — ~512 samples of 235,824 — so a plain\n"
                  "  run reuses the cache silently and scores this seed with the other seed's\n"
                  "  vectors. That is a stale hit, not a miss, and it is the more dangerous of the\n"
                  "  two. Recorded as a finding of its own; not fixed here, because changing a\n"
                  "  cache key at this point forces an unplanned two-hour re-encode.")
        print("  The sweep reuses the cache and **bounds the error instead of hiding it**: the\n"
              "  embeddings row is exact for every work except these, and the held-out overlap is\n"
              "  printed per seed below, which is what caps how far the metric could move.\n")
    return drift, drifted_ids


def check_reference(run: dict[str, object], cached: dict[str, np.ndarray] | None) -> list[str]:
    """Seed 42 must reproduce every published cell and every stored per-user vector.

    Both, not either: four decimals of HitRate is a weaker check than 13,580 booleans, and a
    sweep that measured a moved baseline would be worse than no sweep.
    """
    failures = []
    for name, (rate, coverage, novelty) in run["rows"].items():
        if name not in PUBLISHED:
            continue  # a hybrid rule has no cell in the primary table
        got = (round(rate, 4), round(coverage * 100, 3), round(novelty, 2))
        if got != PUBLISHED[name]:
            failures.append(f"{name}: measured {got}, published {PUBLISHED[name]}")
    for name, vector in run["hits"].items():
        if cached and name in cached and not np.array_equal(vector, cached[name]):
            failures.append(f"{name}: per-user hit vector differs from M18's cached run")
    return failures


def run_seed(seed: int, catalog, *, k: int, tier2: bool, drifted: set[str], fusion_alpha: float) -> dict[str, object]:
    """Fit and score every model on one draw. Returns metrics plus per-user hit vectors."""
    bench = build_bench(work_level=True, catalog=catalog, seed=seed)
    holdout = bench.split.test["ISBN"].to_numpy()
    users = np.sort(bench.split.test["User-ID"].to_numpy())
    # The cap on how far the reused embedding cache could move this seed's embeddings row.
    stale_holdouts = int(np.isin(holdout, list(drifted)).sum()) if drifted else 0
    if drifted:
        print(f"  [cache note] {len(drifted)} works carry stale vectors this seed; "
              f"{stale_holdouts} of {len(holdout):,} held-out books are among them", flush=True)
    rows: dict[str, tuple[float, float, float]] = {}
    hits: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}

    for name in ALL_MODELS:
        started = time.perf_counter()
        try:
            model = fit_model(build_model(name, work_level=True), bench.train, bench.catalog, bench.split.train)
        except SystemExit as exc:
            print(f"  {name:<20} SKIPPED: {exc}", flush=True)
            continue
        fitted[name] = model
        result = evaluate(
            model, bench.split, bench.train, catalog_isbns=bench.catalog_ids, catalog_size=bench.catalog_size, k=k
        )
        rows[name] = (result.hit_rate_at_10, result.coverage_at_10, result.novelty_at_10)
        hits[name] = hit_vector(result.recommendations, holdout)
        print(f"  {name:<20} HitRate@{k}={result.hit_rate_at_10:.4f}  "
              f"Coverage@{k}={result.coverage_at_10:.3%}  Novelty@{k}={result.novelty_at_10:.2f}  "
              f"[{time.perf_counter() - started:.0f}s]", flush=True)

    if tier2 and {"item-item", "tfidf"} <= fitted.keys():
        # RRF first, deliberately: it is **parameter-free**, so it is the one fusion-family row
        # comparable across seeds without inheriting a tuned constant. L78 established it is
        # indistinguishable from tuned fusion (p = 0.867), so the margin-2 question survives
        # through RRF even where the alpha row cannot carry it.
        for label, rule in (("hybrid-rrf", RRF), ("hybrid-cascade", CASCADE), ("hybrid-fusion", FUSION)):
            started = time.perf_counter()
            model = HybridRecommender(
                fitted["item-item"], fitted["tfidf"], rule=rule, alpha=fusion_alpha, name=label
            )
            result = evaluate(
                model, bench.split, bench.train, catalog_isbns=bench.catalog_ids,
                catalog_size=bench.catalog_size, k=k,
            )
            rows[label] = (result.hit_rate_at_10, result.coverage_at_10, result.novelty_at_10)
            hits[label] = hit_vector(result.recommendations, holdout)
            print(f"  {label:<20} HitRate@{k}={result.hit_rate_at_10:.4f}  "
                  f"Coverage@{k}={result.coverage_at_10:.3%}  Novelty@{k}={result.novelty_at_10:.2f}  "
                  f"[{time.perf_counter() - started:.0f}s]", flush=True)

    return {
        "rows": rows,
        "hits": hits,
        "users": users,
        "holdout": holdout,
        "n": len(holdout),
        "stale_holdouts": stale_holdouts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--drift-only", action="store_true", help="run the gate and stop")
    parser.add_argument("--no-tier2", action="store_true", help="skip the three M19 hybrid rules")
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.6,
        help="alpha for the fusion rule. Default 0.6 — L77's value, chosen on the inner validation "
        "split at seed 42 and INHERITED here rather than re-tuned per seed (see the module "
        "docstring). The constructor default is 0.5, which is a different rule and not L77.",
    )
    args = parser.parse_args(argv)
    seeds = tuple(args.seeds)

    started = time.perf_counter()
    catalog = load()
    print(f"catalogue loaded in {time.perf_counter() - started:.0f}s\n", flush=True)

    drift, drifted_ids = drift_gate(seeds, catalog)
    if args.drift_only:
        return 0

    reference = seeds[0]
    cached = None
    if HIT_CACHE.exists() and reference == 42:
        stored = np.load(HIT_CACHE, allow_pickle=False)
        cached = {name: stored[name].astype(bool) for name in ALL_MODELS if name in stored.files}
        print(f"seed 42's published vectors read from {HIT_CACHE} for the bit-for-bit check\n")

    runs: dict[int, dict[str, object]] = {}
    users_reference = None
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        run = run_seed(
            seed, catalog, k=args.k, tier2=not args.no_tier2,
            drifted=drifted_ids[seed], fusion_alpha=args.fusion_alpha,
        )
        # Decision 4: the eligible set is an intersect1d over two data thresholds and the seed
        # enters at a single rng.integers, so every draw must score the SAME readers. That is
        # what makes five runs a paired sample instead of five unrelated experiments. It is
        # documented in split.py and was, until this line, enforced nowhere.
        if users_reference is None:
            users_reference = run["users"]
        elif not np.array_equal(run["users"], users_reference):
            print(f"\nFATAL  seed {seed} scores a different reader set than seed {reference}. "
                  f"The five runs are not a paired sample and nothing below would mean anything.")
            return 1
        if seed == 42 and args.k == 10:
            failures = check_reference(run, cached)
            if failures:
                for line in failures:
                    print(f"REPRODUCTION FAILURE  {line}")
                print("\nSeed 42 no longer reproduces the published table. Stop — the sweep would "
                      "be measuring a moved baseline.")
                return 1
            print("  seed 42 reproduces the published table, all three metrics and every hit "
                  "vector bit-for-bit\n", flush=True)
        runs[seed] = run
        print(flush=True)

    print(f"reader set identical across all {len(seeds)} draws: {len(users_reference):,} readers\n")

    models = [m for m in runs[reference]["rows"]]
    spread = []
    for name in models:
        values = [runs[s]["rows"][name][0] for s in seeds if name in runs[s]["rows"]]
        spread.append(
            {
                "model": name,
                "published": PUBLISHED.get(name, ("—",))[0],
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(statistics.fmean(values), 4),
                "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "range": round(max(values) - min(values), 4),
                "seeds": len(values),
            }
        )
    print("=== HitRate@10 across the draws (levels — the boring half) ===\n")
    print(pd.DataFrame(spread).sort_values("mean", ascending=False).to_markdown(index=False))

    print("\n=== The ordering: paired delta against item-item, per seed (the half that matters) ===\n")
    ordering = []
    for name in models:
        if name == "item-item":
            continue
        row: dict[str, object] = {"model": name}
        signs = set()
        verdicts = set()
        for seed in seeds:
            if name not in runs[seed]["hits"]:
                row[f"seed {seed}"] = "—"
                continue
            test = mcnemar(runs[seed]["hits"][name], runs[seed]["hits"]["item-item"])
            signs.add(np.sign(test["diff"]))
            verdicts.add(test["p"] < 0.05)
            row[f"seed {seed}"] = f"{test['diff']:+.4f} (p={test['p']:.1e})"
        row["sign holds"] = "yes" if len(signs) == 1 else "**NO**"
        row["verdict holds"] = "yes" if len(verdicts) == 1 else "**NO**"
        ordering.append(row)
    print(pd.DataFrame(ordering).to_markdown(index=False))

    print("\n=== The three margins that could plausibly move ===\n")
    margins = list(MARGINS) + ([("hybrid-fusion", "item-item")] if not args.no_tier2 else [])
    detail = []
    for a, b in margins:
        for seed in seeds:
            if a not in runs[seed]["hits"] or b not in runs[seed]["hits"]:
                continue
            test = mcnemar(runs[seed]["hits"][a], runs[seed]["hits"][b])
            detail.append(
                {
                    "margin": f"{a} vs {b}",
                    "seed": seed,
                    "Δ": f"{test['diff']:+.4f}",
                    "|Δ| > 0.004": "yes" if abs(test["diff"]) > 0.004 else "no",
                    "wins": test["a_only"],
                    "losses": test["b_only"],
                    "p": f"{test['p']:.2e}" if test["p"] < 0.001 else f"{test['p']:.3f}",
                    "verdict": "distinguishable" if test["p"] < 0.05 else "NOT distinguishable",
                }
            )
    print(pd.DataFrame(detail).to_markdown(index=False))

    CACHE.mkdir(parents=True, exist_ok=True)
    summary = {
        "seeds": list(seeds),
        "k": args.k,
        "n_users": int(len(users_reference)),
        "canonical_text_drift": drift,
        "embedding_cache_reused": True,
        "fingerprint_noticed_drift": False,
        "stale_holdouts_per_seed": {s: runs[s]["stale_holdouts"] for s in seeds},
        "levels": {row["model"]: {kk: row[kk] for kk in ("min", "max", "mean", "sd", "range")} for row in spread},
        "ordering_against_item_item": ordering,
        "margins": detail,
    }
    out = CACHE / f"seed_sensitivity_work_k{args.k}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    for seed in seeds:
        np.savez_compressed(CACHE / f"hits_work_k{args.k}_seed{seed}.npz", **runs[seed]["hits"])
    print(f"\nwritten to {out}; per-seed hit vectors alongside it")

    # Two different failures, and conflating them raises an alarm the situation does not
    # warrant. A **sign** flip means the table is in the wrong order on some draw — that is
    # what M21's escalation rule is about. A **verdict** change means a near-tie crossed
    # p = 0.05, which is a caveat on one sentence, not a wrong table. Reported separately.
    reordered = [row["model"] for row in ordering if row["sign holds"] == "**NO**"]
    requalified = [row["model"] for row in ordering if row["verdict holds"] == "**NO**"]
    if reordered:
        print(f"\nORDERING FLIPPED for: {', '.join(reordered)} — the table is in a different order "
              f"on some draw. Per M21's escalation rule the primary table is NOT rewritten "
              f"automatically; record the numbers and escalate the decision.")
    else:
        print("\nThe ordering holds on every draw: every comparison keeps its sign, on all "
              f"{len(seeds)} draws.")
    if requalified:
        print(f"Distinguishability is draw-dependent for: {', '.join(requalified)} — same sign "
              f"every time, but p crosses 0.05 between draws. These are near-ties, and the "
              f"caveat belongs on the sentence that calls them separable, not on the ordering.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
