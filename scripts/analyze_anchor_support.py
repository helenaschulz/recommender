"""Is the demo's similarity score comparable across anchors? No — and this measures it (M14.1).

    python scripts/analyze_anchor_support.py
    python scripts/analyze_anchor_support.py --per-band 60 --seed 42
    python scripts/analyze_anchor_support.py --floors 20 50 100 200     # M14.2

The finding this exists to make reproducible: ``similar_min_support = 20`` (ledger L34)
admits anchors whose neighbourhoods rest on almost nothing, and **the similarity score does
not reveal it**. Sample anchors by support band, ask the engine for ten neighbours each, and
count the readers actually shared with the anchor. The evidence column collapses by an order
of magnitude across the bands while the score column stays flat — so a cosine of 0.49 is
*highest* where it is *least* trustworthy.

The mechanism, which is the part worth saying out loud: a factor fitted from 25
interactions is underdetermined and lands in a sparse region of the 128-dimensional space,
where high cosines are cheap. A factor pulled by 700 readers sits in a crowded region where
nothing reaches 0.8. This is L34's argument about *candidates*, applied to the **anchor** —
and ``demo.similar`` already half-anticipates it by refusing anchors under the floor. The
floor is simply set too low, and section 2 prices moving it.

**Measured on the app's own assets**, not on a re-fitted model: the claim is about what a
visitor sees, so it is measured on exactly what a visitor queries. Nothing here is fitted,
so nothing here can be quoted as a model result — same rule as `build_app_assets.py`.

Definitions, pinned here so a re-run means the same thing:

- an **anchor** is an item that has a catalogue row (a visitor can only reach a book the
  app can name) and clears the floor under test;
- **co-readers** of a slot = readers of the anchor who also read that book, i.e. the same
  count the app prints in its reason sentence;
- the **band statistics** are taken over all ``(anchor, slot)`` pairs in the band, not over
  per-anchor summaries — one anchor with two slots and one with ten then carry their true
  weight. The per-anchor median is printed beside it so the two cannot be confused.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from recommender.demo import DemoEngine, load_assets
from recommender.display import THIN_EVIDENCE_SHARE, evidence_share
from recommender.gallery import DEMO_ANCHORS

#: (low, high) interaction counts, high exclusive. The top band is open-ended.
BANDS: list[tuple[int, int]] = [(20, 30), (30, 50), (50, 100), (100, 250), (250, 600), (600, 10**9)]

THIN = 5  # "thin evidence": fewer than this many shared readers behind a shown book


def band_label(low: int, high: int) -> str:
    return f"{low}+" if high >= 10**9 else f"{low}-{high}"


def sample_anchors(rng, support, nameable: np.ndarray, low: int, high: int, n: int) -> np.ndarray:
    """Row indices of up to *n* nameable items whose support is in ``[low, high)``."""
    in_band = np.flatnonzero(nameable & (support >= low) & (support < high))
    if in_band.size <= n:
        return in_band
    return rng.choice(in_band, size=n, replace=False)


def measure(engine: DemoEngine, rows: np.ndarray, k: int) -> dict:
    """Run the engine over *rows* and collect every slot's evidence."""
    co_readers: list[int] = []
    scores: list[float] = []
    shares: list[float] = []
    per_anchor_median: list[float] = []
    empty = 0
    for row in rows.tolist():
        suggestions = engine.similar(str(engine.assets.item_ids[row]), k=k, tau=0)
        if not suggestions:
            empty += 1
            continue
        counts = [s.evidence.co_readers for s in suggestions]
        co_readers.extend(counts)
        scores.extend(s.evidence.score for s in suggestions)
        # The share the app's "thin evidence" tag fires on, measured with the *same* rule
        # the screen uses, so the tag and the ledger cannot drift apart (M15.4).
        shares.extend(evidence_share(s.evidence) for s in suggestions)
        per_anchor_median.append(float(np.median(counts)))
    counts = np.array(co_readers, dtype=float)
    share_array = np.array(shares, dtype=float)
    return {
        "share_tagged": float((share_array < THIN_EVIDENCE_SHARE).mean()) if share_array.size else float("nan"),
        "median_share": float(np.median(share_array)) if share_array.size else float("nan"),
        "n_anchors": len(rows),
        "n_empty": empty,
        "n_slots": counts.size,
        "median_co_readers": float(np.median(counts)) if counts.size else float("nan"),
        "mean_co_readers": float(counts.mean()) if counts.size else float("nan"),
        "median_of_anchor_medians": float(np.median(per_anchor_median)) if per_anchor_median else float("nan"),
        "share_thin": float((counts < THIN).mean()) if counts.size else float("nan"),
        "share_zero": float((counts == 0).mean()) if counts.size else float("nan"),
        "median_score": float(np.median(scores)) if scores else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-band", type=int, default=60)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--floors", type=int, nargs="*", default=[20, 50, 100, 200], help="M14.2")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    # Resident, not memory-mapped: the app pages in whatever one query touches, which is
    # right for a cold start and wrong for a few thousand of them — mmapped, this sweep
    # re-reads 156 MB per anchor. Same arrays, same numbers, three orders of magnitude.
    loaded = load_assets()
    # Section 1 exists to show what the *low* bands look like, so it deliberately reads
    # below the shipped anchor floor. Without this override the shipped floor (50 since
    # M14.2) would silence bands 20-30 and 30-50 — the two bands that are the argument.
    assets = replace(loaded, factors=np.array(loaded.factors), anchor_min_support=BANDS[0][0])
    engine = DemoEngine(assets)
    support = assets.item_support
    # A set membership test, not np.isin: both arrays are object dtype, where isin falls
    # back to an O(n*m) comparison over 305k x 236k and never finishes.
    named = set(assets.books.index)
    nameable = np.array([isbn in named for isbn in assets.item_ids.tolist()])
    print(
        f"assets: {len(assets.item_ids):,} items, {int(nameable.sum()):,} of them nameable, "
        f"shipped floors: candidates {loaded.similar_min_support}, anchors {loaded.anchor_floor} "
        f"(section 1 reads down to {BANDS[0][0]})  [{time.perf_counter() - started:.0f}s]\n",
        flush=True,
    )

    # -- 1 · The calibration finding -------------------------------------------------
    rng = np.random.default_rng(args.seed)
    print("=" * 104)
    print(f"1 · Evidence against score, by anchor support ({args.per_band} random anchors per band, seed {args.seed})")
    print("=" * 104)
    print(
        f"{'anchor support':<16} {'anchors':>8} {'slots':>7} {'median':>8} {'mean':>7} "
        f"{'<5 co-readers':>14} {'<2% of anchor':>14} {'=0':>7} {'median sim':>11}"
    )
    rows_out = []
    for low, high in BANDS:
        rows = sample_anchors(rng, support, nameable, low, high, args.per_band)
        stats = measure(engine, rows, args.k) | {"band": band_label(low, high)}
        rows_out.append(stats)
        print(
            f"{stats['band']:<16} {stats['n_anchors']:>8,} {stats['n_slots']:>7,} "
            f"{stats['median_co_readers']:>8.1f} {stats['mean_co_readers']:>7.1f} "
            f"{stats['share_thin']:>13.1%} {stats['share_tagged']:>13.1%} "
            f"{stats['share_zero']:>6.1%} {stats['median_score']:>11.3f}",
            flush=True,
        )
    top, bottom = rows_out[0], rows_out[-1]
    factor = bottom["median_co_readers"] / max(top["median_co_readers"], 1e-9)
    drift = (bottom["median_score"] - top["median_score"]) / top["median_score"]
    print(
        f"\nacross the bands the median evidence moves {top['median_co_readers']:.1f} -> "
        f"{bottom['median_co_readers']:.1f} co-readers (x{factor:.0f}) while the median similarity "
        f"moves {top['median_score']:.3f} -> {bottom['median_score']:.3f} ({drift:+.0%}), "
        f"in the wrong direction."
    )
    print(
        f"thin slots (<{THIN} co-readers): {top['share_thin']:.1%} in the lowest band, "
        f"{bottom['share_thin']:.1%} in the highest."
    )

    # -- 1b · The self-check M15.4 asks for, on the rule the screen actually uses --------
    tagged = sum(row["share_tagged"] * row["n_slots"] for row in rows_out)
    slots = sum(row["n_slots"] for row in rows_out)
    print(
        f"\n\"thin evidence\" tag (< {THIN_EVIDENCE_SHARE:.0%} of the anchor's readers) fires on "
        f"{tagged / slots:.1%} of all {slots:,} slots — the M15.4 self-check ceiling is 30%."
    )
    print(
        "Read the two thin columns against each other, because they disagree by design and the\n"
        "disagreement is worth knowing: the absolute one falls with support "
        f"({top['share_thin']:.0%} -> {bottom['share_thin']:.0%}) and the share-based one *rises*\n"
        f"({top['share_tagged']:.0%} -> {bottom['share_tagged']:.0%}). At {BANDS[0][0]} readers, "
        f"{THIN_EVIDENCE_SHARE:.0%} of the anchor is under one reader, so a slot cannot be tagged at\n"
        "all; at 900 readers six shared readers is 0.7% and is tagged. The tag is a statement about\n"
        "*this anchor's* audience, not about absolute evidence, and it can only mean that."
    )

    # -- 2 · What a higher floor costs and buys (M14.2) -------------------------------
    print("\n" + "=" * 104)
    print("2 · Pricing the floor: breadth against trustworthiness")
    print("=" * 104)
    total_interactions = int(support.sum())
    nameable_items = int(nameable.sum())
    print(
        "The floor was ONE number until this measurement. It has to be two, and the reason is in\n"
        "the second block: raising the *candidate* floor buys evidence and pays for it in relevance —\n"
        "Dune stops recommending Heretics of Dune, Harry Potter stops recommending Quidditch. L34\n"
        "pinned the candidate floor at 20 and that is still right. L63 is a claim about the ANCHOR.\n"
    )
    print(
        f"{'anchor':>7} {'cand.':>6} {'answerable works':>18} {'share':>8} {'interaction cov.':>18} "
        f"{'thin slots':>12} {'median co-readers':>18}"
    )
    variants = (("anchor floor only (candidates stay at 20)", False), ("both floors together", True))
    for label, raise_candidates in variants:
        print(f"\n  -- {label} " + "-" * (72 - len(label)))
        for floor in args.floors:
            answerable = nameable & (support >= floor)
            # The honest denominator for "how often can we answer a real reader": the share
            # of all interactions that point at a work the engine would still speak about.
            reach = int(support[answerable].sum()) / total_interactions
            rng_floor = np.random.default_rng(args.seed)
            sample = sample_anchors(rng_floor, support, nameable, floor, 10**9, args.per_band * len(BANDS))
            tuned = (
                replace(assets, similar_min_support=floor)
                if raise_candidates
                else replace(assets, anchor_min_support=floor)
            )
            stats = measure(DemoEngine(tuned), sample, args.k)
            print(
                f"{floor:>7} {tuned.similar_min_support:>6} {int(answerable.sum()):>18,} "
                f"{answerable.sum() / nameable_items:>7.1%} {reach:>17.1%} "
                f"{stats['share_thin']:>11.1%} {stats['median_co_readers']:>17.1f}",
                flush=True,
            )
    print(
        "\nThe anchor floor also decides what a visitor may *type*: `find()` offers only works "
        "`similar()` would answer for, so 'answerable works' is the size of the searchable "
        "catalogue, not only of the recommendable one."
    )

    # -- 3 · The same choice, in the only currency a demo is judged in ----------------
    print("\n" + "=" * 104)
    print("3 · Which of the eleven demo anchors each floor would silence")
    print("=" * 104)
    support_of = dict(zip(assets.item_ids.tolist(), support.tolist(), strict=True))
    print(f"{'anchor':<40} {'readers':>8}   " + "  ".join(f"{floor:>5}" for floor in args.floors))
    for work_id, label in DEMO_ANCHORS.items():
        readers = support_of.get(work_id, 0)
        marks = "  ".join(f"{'ok' if readers >= floor else 'GONE':>5}" for floor in args.floors)
        print(f"{label:<40} {readers:>8,}   {marks}")
    print(
        "\nA floor is not only a statistic: it is the list of books the demo can still be asked "
        "about in the room."
    )
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
