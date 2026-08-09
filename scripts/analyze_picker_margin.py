"""How far below the best match is a candidate still worth offering? (M17.4)

    python scripts/analyze_picker_margin.py
    python scripts/analyze_picker_margin.py --sample 400 --seed 7

``DemoEngine.find(query, k=5)`` returns five candidates whatever their score, so
``"little prince"`` offers *A Little Princess*, a John Saul and a Stephen King beside the
right answer. A visibly absurd option damages the credibility of the good ones next to it,
which is a presentation cost, not a metric one — no published number is measured on this
path.

**The margin is derived here rather than chosen**, because a round number picked from three
hand-read queries is exactly how a demo starts lying. The derivation needs a label for
"would a person accept this candidate as a reading of that query", and it gets one without a
human: take a work the engine can actually answer for, **use its own title as the query**,
and call a returned candidate *on-target* when its title contains the query's title or the
query's title contains it. That is a proxy — a translation or a different book with the same
name counts as on-target — and it errs in the safe direction: it inflates the on-target
class, so a margin chosen to keep the on-target class is if anything too generous.

What comes out is a distribution: how far, in cosine, an on-target candidate sits below the
best match, against the same distance for an off-target one. The margin is read off that,
and the script prints both the separation it buys and the picker lengths it produces.

**The floor of the search is :data:`recommender.demo.LOOKUP_TIE_MARGIN`, and that is a
structural constraint, not a preference.** The tie margin re-orders candidates within 0.06
of the best cosine by readership; it is what decides *which book the query resolves to*. Any
picker cutoff at or above 0.06 leaves that group whole, so the resolved anchor cannot move
and the two mechanisms cannot interact. Below 0.06 they would, and M17.4 says to escalate
rather than tune both.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from recommender.data import split_series
from recommender.demo import LOOKUP_TIE_MARGIN, OVERSAMPLE, DemoEngine, load_assets
from recommender.gallery import DEMO_ANCHORS

#: The queries a person actually types in front of the reviewer: the three from M14.9, plus the
#: one that prompted M17. Reported individually, because a distribution cannot show that
#: *A Little Princess* stopped being offered.
LIVE_QUERIES = ["Harry Potter", "Girl with a Pearl Earring", "Guns Germs Steel", "little prince"]

#: Candidate margins to report. The grid starts at the tie margin for the reason in the
#: module docstring and stops where the picker stops being a picker.
GRID = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]


def normalise(title: str) -> str:
    """A title reduced to what a person would type: no parenthetical, no case, no padding."""
    return "".join(ch for ch in split_series(title)[0].casefold() if ch.isalnum() or ch.isspace()).strip()


def shortlist(engine: DemoEngine, query: str, k: int = 5) -> list[tuple[str, str, float, int]]:
    """``find()``'s candidates with the score each one was chosen on, **cutoff not applied**.

    A transcription of :meth:`DemoEngine.find` rather than a call to it, because the scores
    are what this script is about and ``find`` returns :class:`Book` objects without them.
    It deliberately reproduces the path *without* :data:`recommender.demo.PICKER_MARGIN`,
    since the margin is the thing being derived — applying it here would measure the answer
    against itself. Kept in step with ``find`` by hand; the assertion in :func:`main` checks
    the ISBNs agree once ``find`` is asked for the same uncut list.
    """
    assets = engine.assets
    vector = engine._encode(query)  # noqa: SLF001 - this script audits the engine
    scores = np.asarray(assets.lookup_vectors @ vector)
    scores = np.where(assets.lookup_support >= assets.anchor_floor, scores, -np.inf)
    take = min(k * OVERSAMPLE, scores.size)
    best = np.argpartition(-scores, kth=take - 1)[:take]
    best = best[np.argsort(-scores[best], kind="stable")]
    best = best[np.isfinite(scores[best])]
    if best.size == 0:
        return []

    cutoff = scores[best[0]] - LOOKUP_TIE_MARGIN
    tied = [row for row in best.tolist() if scores[row] >= cutoff]
    rest = [row for row in best.tolist() if scores[row] < cutoff]
    tied.sort(key=lambda row: (-assets.lookup_support[row], -scores[row]))

    seen: set[str] = set()
    out: list[tuple[str, str, float, int]] = []
    for row in tied + rest:
        isbn = str(assets.lookup_ids[row])
        if isbn in seen:
            continue
        seen.add(isbn)
        out.append((isbn, engine.describe(isbn).title, float(scores[row]), int(assets.lookup_support[row])))
        if len(out) == k:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=300, help="answerable works used as queries")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=5, help="independent samples, for the argmax spread")
    args = parser.parse_args(argv)

    assets = load_assets()
    engine = DemoEngine(assets)

    # Only works above the anchor floor can be typed for and answered, so only those are
    # sampled: a query for anything else is a lookup failure, not a picker question.
    answerable = np.flatnonzero(assets.lookup_support >= assets.anchor_floor)

    def sweep(seed: int) -> tuple[np.ndarray, np.ndarray, dict[float, list[int]]]:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(answerable, size=min(args.sample, answerable.size), replace=False)
        deltas: list[tuple[float, bool]] = []  # (best score - this score, on target?)
        lengths: dict[float, list[int]] = {margin: [] for margin in GRID}
        for row in chosen.tolist():
            isbn = str(assets.lookup_ids[row])
            query = normalise(engine.describe(isbn).title)
            if not query:
                continue
            candidates = shortlist(engine, query)
            if not candidates:
                continue
            top = max(score for _, _, score, _ in candidates)
            # Rank 1 is excluded from both classes. The query *is* a work's title, so rank 1
            # is on-target by construction and survives every margin — counting it would park
            # a third of the on-target mass at distance 0 and flatter whatever margin came
            # out. The picker question is only ever about the rows *under* the best match.
            for _, title, score, _ in candidates[1:]:
                other = normalise(title)
                on_target = bool(other) and (other in query or query in other)
                deltas.append((top - score, on_target))
            for margin in GRID:
                lengths[margin].append(sum(1 for _, _, score, _ in candidates if score >= top - margin))
        return (
            np.array([delta for delta, hit in deltas if hit]),
            np.array([delta for delta, hit in deltas if not hit]),
            lengths,
        )

    def widest(on: np.ndarray, off: np.ndarray) -> tuple[float, float]:
        """Separation-maximising margin, searched over every distance the sample contains."""
        grid = np.unique(np.concatenate([on, off]))
        grid = grid[grid >= LOOKUP_TIE_MARGIN]
        return max((float(np.mean(on <= m) - np.mean(off <= m)), float(m)) for m in grid)

    on, off, lengths = sweep(args.seed)
    print(f"{args.sample:,} sampled queries, {on.size + off.size:,} alternative slots (rank 1 excluded): "
          f"{on.size:,} on-target, {off.size:,} off-target\n")

    print("distance below the best match (cosine)")
    print(f"{'':<12}{'n':>7}{'p10':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}{'max':>9}")
    for name, values in (("on-target", on), ("off-target", off)):
        if values.size:
            q = np.percentile(values, [10, 25, 50, 75, 90])
            print(f"{name:<12}{values.size:>7}" + "".join(f"{v:>9.3f}" for v in q) + f"{values.max():>9.3f}")

    print("\nwhat each margin keeps")
    print(f"{'margin':>8}{'on-target kept':>16}{'off-target kept':>17}{'separation':>12}"
          f"{'median rows':>13}{'rows=5':>9}{'rows=1':>9}")
    for margin in GRID:
        kept_on = float(np.mean(on <= margin)) if on.size else 0.0
        kept_off = float(np.mean(off <= margin)) if off.size else 0.0
        rows = np.array(lengths[margin])
        print(
            f"{margin:>8.2f}{kept_on:>15.1%}{kept_off:>16.1%}{kept_on - kept_off:>+12.1%}"
            f"{np.median(rows):>13.0f}{np.mean(rows == 5):>9.1%}{np.mean(rows == 1):>9.1%}"
        )

    # The margin, read off the data rather than off the grid: the point of widest separation
    # between the two classes, searched over every distinct distance in the sample — and then
    # repeated on independent samples, because an argmax on a flat curve is a coin toss and
    # reporting one run's third decimal would be a made-up precision.
    print("\nwidest separation over all observed distances >= the tie margin")
    peaks = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        gain, margin = widest(on, off) if repeat == 0 else widest(*sweep(seed)[:2])
        peaks.append(margin)
        print(f"  seed {seed:<4} {gain:+.1%} at {margin:.3f}")
    print(f"  {args.repeats} independent samples: median {np.median(peaks):.3f}, "
          f"range {min(peaks):.3f}-{max(peaks):.3f}")

    settled = float(np.round(np.median(peaks), 2))
    print(f"\n=== reporting the rest at a margin of {settled:.2f} ===")

    print("\nthe typed live-demo queries, in full. 'drop' is what the picker stops offering.")
    for query in LIVE_QUERIES:
        candidates = shortlist(engine, query)
        # The transcription above must be the engine's own path, or every number here is
        # about a function the app does not call.
        uncut = engine.find(query, k=5, margin=2.0)  # 2.0 exceeds any possible cosine gap
        assert [isbn for isbn, _, _, _ in candidates] == [book.isbn for book in uncut]
        top = max((score for _, _, score, _ in candidates), default=0.0)
        print(f"\n  {query!r}")
        for rank, (_, title, score, readers) in enumerate(candidates, start=1):
            verdict = "keep" if score >= top - settled else "drop"
            print(f"    {rank}. {verdict}  {score:.3f}  (-{top - score:.3f})  "
                  f"{title[:60]:<60} {readers:>5} readers")

    print("\nthe eleven anchors, typed as their own titles: picker rows before -> after")
    for work_id, label in DEMO_ANCHORS.items():
        query = normalise(engine.describe(work_id).title)
        candidates = shortlist(engine, query)
        top = max((score for _, _, score, _ in candidates), default=0.0)
        kept = [c for c in candidates if c[2] >= top - settled]
        gaps = " ".join(f"{top - score:.3f}" for _, _, score, _ in candidates[1:])
        print(f"  {label:<38} {len(candidates)} -> {len(kept)}  "
              f"rank1={(candidates[0][1][:34] if candidates else '-'):<36} gaps: {gaps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
