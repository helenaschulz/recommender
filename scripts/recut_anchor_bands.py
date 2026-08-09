"""Re-cut the anchor table's support bands at the boundary the floor decision turns on (M23.2).

    python scripts/recut_anchor_bands.py                    # both draws, both cuts
    python scripts/recut_anchor_bands.py --seeds 42         # the pinned draw only

**The question.** L81 and L85 both report the same shape: every engine this project has
measured is separated from item-item **below** the app's anchor floor and indistinguishable
**above** it. M23 asks whether the floor should move from 50 to 20 — and neither line can
answer that, because both stop at a ``5-49`` band that mixes 5-19 with 20-49. The band that
would actually open is **20-49**, and L85's headline 58/7 could be carried entirely by the
half that stays closed. This re-cuts the identical vectors at 5 / 20 / 50.

**Nothing is re-scored and nothing is re-fitted.** It reads the per-anchor hit vectors that
``scripts/measure_anchor_hitrate.py`` cached (``artifacts/anchors/``) and groups them, so the
hit is :func:`recommender.eval.hit_vector` exactly as in L80, L81 and L85 — the M20 and M22
rule, kept. The published boundaries are printed **beside** the new ones rather than replaced,
because every existing row has to stay readable against them.

**Both draws, always.** L81 is the standing warning here: its 50+ band read "not
distinguishable" at seed 42 and "item-item wins, p = 0.002" at seed 43, and a milestone that
quotes one draw of a stratum is quoting the friendlier one. Two draws are not a distribution,
so a band whose two draws disagree is reported as unsettled rather than resolved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.eval import FLOOR_BANDS, PUBLISHED_BANDS, anchor_strata_pairwise, anchor_strata_table

CACHE = Path("artifacts/anchors")

#: The rows the floor decision is actually about — M23's three configurations (A: ALS,
#: B: item-item, C: RRF) plus the rule the app would be replacing.
CONFIGURATIONS = ("als", "item-item", "hybrid (rrf)", "hybrid (cascade)", "hybrid (fusion)", "content-tfidf")


def load(seed: int, k: int) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    hits_file = CACHE / f"anchor_hits_work_k{k}_seed{seed}.npz"
    draw_file = CACHE / f"anchors_work_seed{seed}.npz"
    if not hits_file.exists() or not draw_file.exists():
        print(f"note: seed {seed} not cached ({hits_file}) — run measure_anchor_hitrate.py first")
        return None
    stored = np.load(hits_file, allow_pickle=False)
    hits = {name: stored[name].astype(bool) for name in stored.files}
    support = np.load(draw_file, allow_pickle=False)["support"]
    return hits, support


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43])
    parser.add_argument("--against", nargs="*", default=["item-item", "als"])
    args = parser.parse_args(argv)

    seen = 0
    for seed in args.seeds:
        loaded = load(seed, args.k)
        if loaded is None:
            continue
        seen += 1
        hits, support = loaded
        print("=" * 100)
        print(f"anchor draw seed {seed} — {len(support):,} anchors, {len(hits)} rows")
        print("=" * 100)

        for title, bands in (("the published cut (L73/L81/L85)", PUBLISHED_BANDS), ("the M23 re-cut", FLOOR_BANDS)):
            print(f"\nAnchorHitRate@{args.k} by the anchor's train support — {title}\n")
            print(anchor_strata_table(hits, support, bands).to_markdown(index=False))

        for opponent in args.against:
            if opponent not in hits:
                continue
            print(f"\nPaired McNemar against {opponent}, re-cut at 5 / 20 / 50 (wins/losses)\n")
            table = anchor_strata_pairwise(hits, support, opponent, FLOOR_BANDS)
            rows = table[table["model"].isin(CONFIGURATIONS)]
            print((rows if not rows.empty else table).to_markdown(index=False))

        counts = pd.DataFrame(
            [
                {
                    "band": label,
                    "anchors": int(mask.sum()),
                    "share of anchors": f"{mask.mean():.1%}",
                    "share of the askable ones (>=20)": f"{mask.sum() / max((support >= 20).sum(), 1):.1%}"
                    if label.startswith(("20-49", "50+"))
                    else "—",
                }
                for label, mask in (
                    (lbl, (support >= lo) if hi is None else ((support >= lo) & (support <= hi)))
                    for lbl, lo, hi in FLOOR_BANDS
                )
            ]
        )
        print("\nWhat each band is worth in anchors\n")
        print(counts.to_markdown(index=False))

    if not seen:
        print("FATAL  no cached draw found — nothing was measured")
        return 1
    print(
        "\nRead the 20-49 column first: it is the band a floor of 20 would open, and it is the\n"
        "only column in this table that the demo's floor decision can act on. A rule that wins\n"
        "5-19 and loses 20-49 wins L85's aggregate and buys the app nothing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
