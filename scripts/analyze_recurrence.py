"""Is the demo just returning bestsellers? Measure it rather than assert it (M14.7).

    python scripts/analyze_recurrence.py
    python scripts/analyze_recurrence.py --anchors 500 --seed 42

"How do you know this isn't a popularity ranking with extra steps" is the question the
panel will ask, and until now the answer was an anecdote: across the eleven anchors Helena
read, exactly one title recurred (*The Secret Life of Bees*, under *The Lovely Bones* and
*Girl with a Pearl Earring* — two book-club literary novels, which is a reason rather than
a coincidence). Eleven anchors is not a measurement. This generalises it.

The design is a contrast, because a recurrence count on its own means nothing without
knowing what the degenerate answer would look like:

- **the demo**: N random answerable anchors, top-10 each, and the distribution of how often
  each work appears across those N lists;
- **pure popularity on the same anchors**: rank the same candidate pool by interaction
  count, exclude the anchor itself. That recommender puts (almost) the *same* ten books in
  every list, so its recurrence is ~100% by construction and its distinct-title count is
  ~10 — the ceiling of the failure mode being ruled out.

Computed from ``item_support`` rather than by fitting ``models/popularity.py``: the app is
fitted on the full matrix and the evaluation baseline on ``split.train``, so re-using the
fitted baseline here would silently mix two universes. Popularity on this pool *is* a sort
by interaction count, and doing it on the app's own numbers keeps both columns comparable.
**Nothing here is fitted and nothing here is a model result** — same rule as the other app
scripts.

One thing this measurement cannot do, stated up front: the candidate pool is already the
7,523 works above the support floor, i.e. the top 3.2% of the nameable catalogue. So "share
of slots in the global top 1% of works" is a soft bar here and is reported next to the pool
size rather than on its own.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import replace

import numpy as np

from recommender.demo import DemoEngine, load_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    loaded = load_assets()
    assets = replace(loaded, factors=np.array(loaded.factors))
    engine = DemoEngine(assets)
    support = assets.item_support
    named = set(assets.books.index)
    nameable = np.array([isbn in named for isbn in assets.item_ids.tolist()])
    # The **anchor** floor, not the candidate floor: this samples things a visitor could
    # actually ask about, and since M14.2 those are two different numbers.
    answerable = np.flatnonzero(nameable & (support >= assets.anchor_floor))

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(answerable, size=min(args.anchors, answerable.size), replace=False)
    print(
        f"{len(rows):,} random anchors from the {answerable.size:,} askable works "
        f"(anchor floor {assets.anchor_floor}, candidate floor {assets.similar_min_support}), "
        f"top-{args.k} each, seed {args.seed}\n",
        flush=True,
    )

    # -- the demo -------------------------------------------------------------------
    demo_counts: Counter[str] = Counter()
    answered: list[int] = []
    for row in rows.tolist():
        suggestions = engine.similar(str(assets.item_ids[row]), k=args.k, tau=0)
        if not suggestions:
            continue
        answered.append(row)
        demo_counts.update(s.isbn for s in suggestions)
    lists = len(answered)

    # -- pure popularity on the same pool and the same anchors -----------------------
    # Over `answered`, not over `rows`: the two columns have to be built from the same
    # lists, or the recurrence shares are divided by different denominators.
    candidates = np.flatnonzero(nameable & (support >= assets.similar_min_support))
    ranked = candidates[np.argsort(-support[candidates], kind="stable")]
    pop_counts: Counter[str] = Counter()
    for row in answered:
        top = [r for r in ranked.tolist() if r != row][: args.k]
        pop_counts.update(str(assets.item_ids[r]) for r in top)

    def describe(counts: Counter[str], label: str) -> None:
        slots = sum(counts.values())
        top_share = counts.most_common(1)[0][1] / lists if counts else 0.0
        top100 = sum(count for _, count in counts.most_common(100))
        print(f"{label:<22} {len(counts):>10,} {slots:>9,} {top_share:>13.1%} {top100 / slots:>15.1%}")

    print(f"{'':<22} {'distinct':>10} {'slots':>9} {'top title in':>13} {'top 100 works':>15}")
    print(f"{'':<22} {'works':>10} {'filled':>9} {'x of lists':>13} {'take x of slots':>15}")
    describe(demo_counts, "the demo (ALS)")
    describe(pop_counts, "pure popularity")

    print("\nthe ten most-recurring works in the demo's answers:")
    for isbn, count in demo_counts.most_common(10):
        row = assets.books.loc[isbn] if isbn in assets.books.index else None
        title = str(row["Book-Title"])[:58] if row is not None else isbn
        print(f"  {count:>4} of {lists:,} lists ({count / lists:>5.1%})  {title}")

    once = sum(1 for count in demo_counts.values() if count == 1)
    print(
        f"\n{once:,} of {len(demo_counts):,} works ({once / len(demo_counts):.1%}) appear in exactly one list. "
        f"A popularity ranking has no such works at all."
    )

    # The soft bar, reported with the caveat attached rather than in a footnote.
    top_one_percent = set(
        str(assets.item_ids[r]) for r in np.argsort(-support, kind="stable")[: max(1, len(support) // 100)].tolist()
    )
    demo_slots = sum(demo_counts.values())
    in_top = sum(count for isbn, count in demo_counts.items() if isbn in top_one_percent)
    print(
        f"\nshare of the demo's slots that sit in the global top 1% of works by interactions: "
        f"{in_top / demo_slots:.1%} — read against the fact that the candidate pool is already the "
        f"top {candidates.size / int(nameable.sum()):.1%} of the nameable catalogue, and against ALS's "
        f"95.7% at ISBN level in ledger L33."
    )
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
