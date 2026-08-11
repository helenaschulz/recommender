"""Write the inspectable table for the face-validity gate on the band a lower floor opens (M23.3).

    python scripts/audit_floor_band.py                       # docs/floor_band_audit.md
    python scripts/audit_floor_band.py --anchors 12 --seed 7

**Why this is a gate and not a footnote (M23 decision 4).** L79 hand-counted the same
item-to-item surface *above* the floor and found cascade 0 bad slots of 30, fusion 11, RRF
13. The rule with the best predictive column had the worst-looking lists. M23 proposes
opening the 20–49 band, where every engine has less evidence per slot than it does above 50
— so the question "do these lists look sensible" has to be answered on **that** band, for
each configuration, before any switcher is built.

**What this script does and does not do.** It samples anchors from the newly-opened band,
asks each configuration for ten neighbours, and writes them side by side with the evidence
the app would print. It does **not** score them: the counting is done by a human reading the
table, exactly as L42 and L79 were counted, and the count is recorded in the ledger with the
counter named. A script that graded its own output here would be measuring a title-matching
rule, not face validity — and L47 is the standing proof that the title key cannot see a
translation.

**The criterion is L79's, inherited unchanged so the two counts are comparable.** A slot is
bad when it is:

1. the anchor itself under another edition, title or translation;
2. a companion or "about" book rather than a comparable read;
3. a duplicate of another slot in the same list.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from recommender.demo import DemoEngine, load_assets
from recommender.engines import build_source

OUT = Path("docs/floor_band_audit.md")

#: The three configurations of M23, named as the milestone names them.
CONFIGURATIONS: tuple[tuple[str, str], ...] = (
    ("A", "als"),
    ("B", "item-item"),
    ("C", "rrf"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=int, default=8, help="how many anchors to sample from the band")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low", type=int, default=20, help="the floor M23 proposes")
    parser.add_argument("--high", type=int, default=49, help="the last support the shipped floor excludes")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    loaded = load_assets()
    assets = replace(loaded, factors=np.array(loaded.factors), anchor_min_support=args.low)
    support = assets.item_support
    named = set(assets.books.index)
    nameable = np.array([isbn in named for isbn in assets.item_ids.tolist()])

    in_band = np.flatnonzero(nameable & (support >= args.low) & (support <= args.high))
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(in_band, size=min(args.anchors, in_band.size), replace=False)
    picked = picked[np.argsort(-support[picked])]

    engines = {}
    for label, name in CONFIGURATIONS:
        engines[label] = DemoEngine(assets, source=build_source(name, assets))
    print(f"engines built in {time.perf_counter() - started:.0f}s", flush=True)

    lines = [
        "# Face-validity audit of the 20–49 band (milestone M23.3)",
        "",
        f"`python scripts/audit_floor_band.py --anchors {args.anchors} --seed {args.seed}` — "
        f"{len(picked)} anchors drawn uniformly from the **{args.low}–{args.high} reader** band, "
        f"the band the shipped anchor floor of 50 closes and a floor of {args.low} would open. "
        f"{len(picked)} anchors x 3 configurations x {args.k} slots = "
        f"**{len(picked) * 3 * args.k} slots** to read.",
        "",
        "Configurations: **A** = ALS (what ships), **B** = item-item shrunk cosine, "
        "**C** = RRF over item-item + TF-IDF. The candidate floor stays at 20 in all three "
        "(L34); only the engine changes.",
        "",
        "**Count a slot bad when it is** (L79's criterion, inherited unchanged so the two "
        "counts are comparable): the anchor itself under another edition, title or "
        "translation; a companion or *about* book rather than a comparable read; or a "
        "duplicate of another slot in the same list.",
        "",
        "`co` is the number of the anchor's readers who also read that book — the number the "
        "app prints. `score` is each engine's own quantity and is **not comparable across "
        "columns**: A is an ALS factor cosine, B a shrunk cosine, C a fused rank sum.",
        "",
    ]

    for row in picked.tolist():
        anchor_id = str(assets.item_ids[row])
        meta = engines["A"].describe(anchor_id)
        lines += [
            f"## {meta.title} by {meta.author}",
            "",
            f"`{anchor_id}` · **{int(support[row]):,} readers**",
            "",
            "| # | A · ALS | co | B · item-item | co | C · RRF | co |",
            "|---|---|---:|---|---:|---|---:|",
        ]
        answers = {label: engine.similar(anchor_id, k=args.k, tau=0) for label, engine in engines.items()}
        for rank in range(args.k):
            cells = []
            for label, _ in CONFIGURATIONS:
                got = answers[label]
                if rank < len(got):
                    s = got[rank]
                    title = s.title.replace("|", "\\|")
                    cells += [f"{title} by *{s.author}*", f"{s.evidence.co_readers:,}"]
                else:
                    cells += ["—", "—"]
            lines.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## The count",
        "",
        "Filled in by hand after reading the table above. Empty until someone has actually "
        "read it — a count written by the script that produced the lists would be the thing "
        "this audit exists to avoid.",
        "",
        "| configuration | bad slots | of | counted by | date |",
        "|---|---:|---:|---|---|",
        f"| A · ALS | | {len(picked) * args.k} | | |",
        f"| B · item-item | | {len(picked) * args.k} | | |",
        f"| C · RRF | | {len(picked) * args.k} | | |",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    slots = len(picked) * 3 * args.k
    print(f"wrote {OUT} — {len(picked)} anchors, {slots} slots  [{time.perf_counter() - started:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
