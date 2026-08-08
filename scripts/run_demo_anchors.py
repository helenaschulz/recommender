"""The demo's answer for a fixed anchor set, as pasteable markdown (M14).

    python scripts/run_demo_anchors.py
    python scripts/run_demo_anchors.py --anchors "fight club|palahniuk" --k 10

Reading the *output* is what found M14 at all: the comparison table has no cell for "the
author tag is missing", "rank 1 and rank 2 are the same book" or "slot 9 is noise". This
script makes that reading reproducible, so the before/after of the M14 fixes is a diff
rather than a memory.

The anchor set and the reasoning behind it live in
:data:`recommender.gallery.DEMO_ANCHORS`, so the sweep script and this one cannot drift
apart about what "the eleven anchors" means. They are addressed by **work id**, not by
ISBN, because the app's item is a work (M12); a missing id is an error rather than a
silently shorter table.
"""

from __future__ import annotations

import argparse
import sys

from recommender.data import split_series
from recommender.demo import DemoEngine, load_assets
from recommender.gallery import DEMO_ANCHORS


def render(engine: DemoEngine, work_id: str, label: str, k: int) -> str:
    """One markdown table for one anchor: rank, book, and every piece of evidence."""
    anchor = engine.describe(work_id)
    lines = [
        f"#### {label} — {anchor.readers:,} readers",
        "",
        "| # | book | author | co-readers | same author | series | sim |",
        "|--:|---|---|--:|:-:|---|--:|",
    ]
    suggestions = engine.similar(work_id, k=k)
    if not suggestions:
        lines.append("| — | *below the support floor: the engine refuses this anchor* | | | | | |")
    for rank, item in enumerate(suggestions, start=1):
        title = split_series(item.title)[0]
        evidence = item.evidence
        lines.append(
            f"| {rank} | {title} | {item.author} | {evidence.co_readers} | "
            f"{'yes' if evidence.same_author else ''} | {item.series} | {evidence.score:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", nargs="*", default=None, help="work ids; default: the eleven")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--reasons", action="store_true", help="also print the reason sentences")
    args = parser.parse_args(argv)

    assets = load_assets()
    engine = DemoEngine(assets)
    anchors = {work: DEMO_ANCHORS.get(work, work) for work in args.anchors} if args.anchors else DEMO_ANCHORS

    missing = [work for work in anchors if work not in assets.item_index]
    if missing:
        print(f"not in the assets: {missing}", file=sys.stderr)
        return 1

    print(f"### The {len(anchors)} anchors, top-{args.k} each")
    print(f"\nassets: {len(assets.item_ids):,} items, support floor {assets.similar_min_support}\n")
    for work_id, label in anchors.items():
        print(render(engine, work_id, label, args.k))
        if args.reasons:
            for rank, item in enumerate(engine.similar(work_id, k=args.k), start=1):
                print(f"    {rank:>2}. {item.reason}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
