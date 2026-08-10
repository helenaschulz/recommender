"""Configuration A is still the engine that shipped, checked against first principles (M23.10.6).

    python scripts/verify_configuration_a.py
    python scripts/verify_configuration_a.py --decimals 12

M23 turned one hard-wired dot product into a seam and M23.10 put a picker on top of it. The
whole thing rests on one claim: **switching the engine changes the list and nothing else, and
not switching it changes nothing at all.** If the default path drifted, every published demo
number and the eleven rehearsed anchors would drift with it — silently, because a
recommendation list has no test that fails.

**There is no stored baseline, on purpose.** A golden file would have to be regenerated from
the gitignored 894 MB assets, and a baseline nobody can rebuild is a baseline nobody checks.
Instead this script re-derives what ``similar`` returned *before the seam existed*, from the
arrays, the obvious way — ``factors @ factors[anchor]``, the L34 candidate floor, a stable
argsort, drop what the catalogue cannot name (L46), take ten — and compares three things
against what the engine returns today:

1. the ten work ids, in order;
2. their scores to ``--decimals`` places, twelve by default;
3. the co-reader count, the same-author tag and the reason sentence on every slot.

Then it does it again through the **new** constructor argument, ``configuration="A"``, so that
"the default did not move" and "the named configuration is the default" are two checks rather
than one assumption.

Eleven anchors x ten slots = **110 slots**, which is the comparison M23's Tier 1 ran by hand
and this script makes reproducible. It exits non-zero on any difference and prints the slot.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from recommender.data import split_series
from recommender.demo import DemoEngine, Evidence, load_assets, reason_sentence, same_author
from recommender.gallery import DEMO_ANCHORS


def first_principles(assets, work_id: str, k: int) -> list[tuple[str, float, int, bool, str]]:
    """What ``DemoEngine.similar`` returned in M22, recomputed without touching the seam."""
    index = assets.item_index
    anchor = index[work_id]
    factors = np.asarray(assets.factors)
    scores = np.asarray(factors @ factors[anchor], dtype=np.float64)
    scores[assets.item_support < assets.similar_min_support] = -np.inf
    scores[anchor] = -np.inf
    order = np.argsort(-scores, kind="stable")

    readers = assets.readers
    def readers_of(row: int) -> np.ndarray:
        lo, hi = readers.indptr[row], readers.indptr[row + 1]
        return readers.indices[lo:hi]

    anchor_readers = readers_of(anchor)
    anchor_meta = DemoEngine(assets).describe(work_id)
    anchor_title = split_series(anchor_meta.title)[0]

    out: list[tuple[str, float, int, bool, str]] = []
    seen = {work_id}
    for row in order.tolist():
        other = str(assets.item_ids[row])
        if other in seen or other not in assets.books.index or not np.isfinite(scores[row]):
            continue
        seen.add(other)
        book = DemoEngine(assets).describe(other)
        evidence = Evidence(
            score=float(scores[row]),
            co_readers=int(np.intersect1d(anchor_readers, readers_of(row), assume_unique=True).size),
            anchor_readers=int(anchor_readers.size),
            same_author=same_author(book.author, anchor_meta.author),
        )
        out.append(
            (other, evidence.score, evidence.co_readers, evidence.same_author,
             reason_sentence(evidence, anchor_title))
        )
        if len(out) == k:
            break
    return out


def as_rows(suggestions, decimals: int) -> list[tuple[str, str, int, bool, str]]:
    return [
        (s.isbn, f"{s.evidence.score:.{decimals}f}", s.evidence.co_readers, s.evidence.same_author, s.reason)
        for s in suggestions
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decimals", type=int, default=12)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    assets = load_assets()
    default = DemoEngine(assets)
    named = DemoEngine(assets, configuration="A")
    print(
        f"assets loaded in {time.perf_counter() - started:.1f}s · "
        f"default source {type(default.source).__name__} · "
        f"configuration A source {type(named.source).__name__}",
        flush=True,
    )

    failures = 0
    slots = 0
    for work_id, label in DEMO_ANCHORS.items():
        if work_id not in assets.item_index:
            print(f"MISSING  {work_id}", file=sys.stderr)
            failures += 1
            continue
        want = [
            (isbn, f"{score:.{args.decimals}f}", co, tag, reason)
            for isbn, score, co, tag, reason in first_principles(assets, work_id, args.k)
        ]
        got_default = as_rows(default.similar(work_id, k=args.k), args.decimals)
        got_named = as_rows(named.similar(work_id, k=args.k), args.decimals)
        slots += len(want)

        for name, got in (("default", got_default), ("configuration=A", got_named)):
            if got == want:
                continue
            failures += 1
            print(f"\nDIFFERS  {label}  ({name})", file=sys.stderr)
            for rank, (a, b) in enumerate(zip(want, got, strict=False), start=1):
                if a != b:
                    print(f"  slot {rank}\n    pre-seam: {a}\n    now     : {b}", file=sys.stderr)
            if len(want) != len(got):
                print(f"  lengths differ: {len(want)} against {len(got)}", file=sys.stderr)
        print(f"  {label:<40} {len(want):>2} slots  ok", flush=True)

    print(
        f"\n{len(DEMO_ANCHORS)} anchors, {slots} slots, scores to {args.decimals} decimals, "
        f"co-reader counts and reason sentences compared  [{time.perf_counter() - started:.0f}s]"
    )
    if failures:
        print(f"{failures} comparisons differ -- configuration A has moved", file=sys.stderr)
        return 1
    print("configuration A is byte-identical to the pre-seam engine, through both entry points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
