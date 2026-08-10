"""Build one configuration's precomputed answer table beside the shipped assets (M23.10.1).

    python scripts/build_answer_table.py                    # configuration B
    python scripts/build_answer_table.py --engine item-item --k 10

**Serving, not evaluation, and that distinction is the first thing to say.** Every row here is
computed on the **full** work-keyed interaction matrix, the same matrix
``scripts/build_app_assets.py`` fits the ALS factors on and for the same reason: the app
computes no metric, so withholding a reader's interactions would make the product worse for
nothing. Nothing in this file may be read as a measurement, and no published number is.

**Additive by construction (M23 decision 7).** This script *reads* ``artifacts/app/*`` and
writes exactly one new file beside it. It never rewrites an asset, so configuration A's
answers cannot move because it ran — which is the property ``scripts/verify_configuration_a.py``
checks independently, on the eleven rehearsed anchors, to twelve decimals.

**The table stores what the engine kept, not what the source ranked.** Rows are taken from
:meth:`recommender.demo.DemoEngine.similar`, i.e. after the nameability filter (L46) and the
candidate floor (L34), so a stored row is a slot the app would actually print. That is what
makes it 2,508 x 10 rather than 2,508 x 100, and it means the depth is spent once at build
time instead of on every query.

**It verifies itself against a live recomputation before it reports anything.** The table is
written, re-read from disk, and then every anchor is answered twice — once from the table,
once from the live source — and the two lists are compared slot by slot, with the worst score
deviation reported rather than assumed. A cache whose equality with the thing it caches is
asserted in the same run is a cache you can put in a serving path; one that is merely believed
is not.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from recommender.answers import askable_rows, load_answers, save_answers, table_path
from recommender.demo import DemoEngine, load_assets
from recommender.engines import CONFIGURATIONS, build_source

#: What the app asks for. Stored rather than assumed, because a table built at k=10 cannot
#: answer a k=20 query and the reader of this file should not have to work that out.
DEFAULT_K = 10


def answers_for(engine: DemoEngine, assets, rows: np.ndarray, k: int) -> tuple[list[list[int]], list[list[float]]]:
    """Ask *engine* for every anchor's top-k, as row indices into the asset arrays."""
    index = assets.item_index
    all_rows: list[list[int]] = []
    all_scores: list[list[float]] = []
    for anchor in rows.tolist():
        suggestions = engine.similar(str(assets.item_ids[anchor]), k=k)
        all_rows.append([index[s.isbn] for s in suggestions])
        all_scores.append([s.evidence.score for s in suggestions])
    return all_rows, all_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="B", choices=sorted(CONFIGURATIONS))
    parser.add_argument("--engine", default=None, help="override the configuration's engine name")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=None, help="directory; default: the assets' own")
    parser.add_argument("--no-verify", action="store_true", help="skip the live re-comparison (not advised)")
    args = parser.parse_args(argv)

    config = CONFIGURATIONS[args.configuration]
    engine_name = args.engine or config.engine

    started = time.perf_counter()
    assets = load_assets()
    directory = args.out or assets.directory
    loaded = time.perf_counter() - started
    print(
        f"assets: {len(assets.item_ids):,} items, work key {assets.work_key!r}, "
        f"anchor floor {assets.anchor_floor}, candidate floor {assets.similar_min_support}  "
        f"[{loaded:.1f}s]",
        flush=True,
    )

    anchors = askable_rows(assets)
    print(f"askable anchors: {anchors.size:,}", flush=True)

    build_started = time.perf_counter()
    live = DemoEngine(assets, source=build_source(engine_name, assets))
    rows, scores = answers_for(live, assets, anchors, args.k)
    build_seconds = time.perf_counter() - build_started

    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows])
    flat_rows = np.array([r for row in rows for r in row], dtype=np.int32)
    flat_scores = np.array([s for row in scores for s in row], dtype=np.float32)

    path = save_answers(
        table_path(directory, engine_name),
        engine=engine_name,
        score_label=config.score_label,
        anchors=anchors,
        indptr=indptr,
        rows=flat_rows,
        scores=flat_scores,
        assets=assets,
        meta={
            "configuration": config.key,
            "k": int(args.k),
            "n_anchors": int(anchors.size),
            "n_rows": int(flat_rows.size),
            "build_seconds": round(build_seconds, 2),
            "built_at": date.today().isoformat(),
            "fitted_on": "the full work-keyed interaction matrix (serving, not evaluation)",
            "askable": "item_support >= anchor_min_support and nameable by the catalogue (L65)",
        },
    )
    size = path.stat().st_size
    short = [i for i, r in enumerate(rows) if len(r) < args.k]
    print(
        f"wrote {path}\n"
        f"  {anchors.size:,} anchors x up to {args.k} = {flat_rows.size:,} rows, "
        f"{size / 1e6:.2f} MB ({size / max(flat_rows.size, 1):.1f} bytes a row)\n"
        f"  built in {build_seconds:.1f}s ({build_seconds / max(anchors.size, 1) * 1000:.0f} ms an anchor)\n"
        f"  anchors with fewer than {args.k} slots: {len(short)}",
        flush=True,
    )

    if args.no_verify:
        return 0

    # -- the table is only worth what its equality with the live source is worth ----------
    verify_started = time.perf_counter()
    table = load_answers(directory, engine_name, assets)
    served = DemoEngine(assets, source=table)
    # Hoisted, because `assets.item_index` is a property that rebuilds a 304,001-entry dict
    # on every access -- reading it inside the loop turned a 40-second verification into one
    # that had not finished in ten minutes.
    index = assets.item_index
    mismatched = 0
    worst = 0.0
    worst_at = ""
    for position, anchor in enumerate(anchors.tolist()):
        work_id = str(assets.item_ids[anchor])
        got = served.similar(work_id, k=args.k)
        want_rows, want_scores = rows[position], scores[position]
        if [index[s.isbn] for s in got] != want_rows:
            mismatched += 1
            continue
        for slot, (s, want) in enumerate(zip(got, want_scores, strict=True)):
            deviation = abs(s.evidence.score - want)
            if deviation > worst:
                worst, worst_at = deviation, f"{work_id} slot {slot + 1}"
    verify_seconds = time.perf_counter() - verify_started
    print(
        f"verified against the live source in {verify_seconds:.1f}s\n"
        f"  anchors whose list differs: {mismatched}\n"
        f"  worst score deviation: {worst:.3e}" + (f"  ({worst_at})" if worst_at else ""),
        flush=True,
    )
    if mismatched:
        print("the table does not reproduce the live ranking -- not usable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
