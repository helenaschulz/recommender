"""What the trained system actually costs to serve — the Part 3 sizing numbers.

Part 3 argues an architecture. An architecture argued without sizes is a box diagram, and
a reviewing architect will ask for the sizes. This script produces them, so the answer is
a measurement rather than an estimate made at the whiteboard.

Two kinds of number come out, and the distinction is kept visible in the output because it
is the difference between evidence and arithmetic:

- **MEASURED** — bytes on disk of the assets the demo actually serves from
  (``artifacts/app/``), plus the shapes and dtypes inside them. This is the shipped system,
  not a model of it.
- **DERIVED** — footprints of structures the serving path would hold in a production
  design, computed from the pinned dimensions (item count, factor count, neighbour count).
  Arithmetic on measured inputs, labelled as such. Nothing here is fitted or timed.

The headline the numbers support: **the demo ships 890 MB and a production serving tier
needs about 1.5 MB of it.** Almost the entire footprint is the free-text lookup (the
sentence-encoder vectors) and an id column stored as fixed-width Unicode. Neither is the
recommender.

Run: ``python scripts/measure_serving_footprint.py``
Reproduces ledger lines L71 and L72. Reads only; writes nothing; fits nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Pinned dimensions, so the derived rows can be recomputed without loading anything.
# Sources: L49 (train matrix), L53 (item-item neighbours), L55 (ALS factors), meta.json.
ITEM_ITEM_NEIGHBOURS = 50  # L53 / L25
ALS_FACTORS = 128  # L55
TOP_N = 10  # every published metric is @10
#: The five-engine shortlist L72 sizes: item-item, ALS, TF-IDF, embeddings, popularity.
#: It is a sizing bound for the whole product, not a description of the shipped app —
#: that offers two configurations (M23.10). The five stay because L72 is published on them.
ENGINES = 5


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir():
            return parent
    return here.parents[1]


def human(n: int) -> str:
    """Bytes in MB, decimal — the unit a storage bill is written in."""
    return f"{n / 1e6:,.1f} MB"


def measure_assets(assets: Path) -> dict:
    """MEASURED: every file the demo loads, byte for byte."""
    rows = []
    total = 0
    for path in sorted(assets.iterdir()):
        if path.name.startswith("."):
            continue
        size = path.stat().st_size
        total += size
        detail = ""
        if path.suffix == ".npy":
            try:
                arr = np.load(path, allow_pickle=True, mmap_mode="r")
            except ValueError:
                arr = np.load(path, allow_pickle=True)
            detail = f"{arr.shape} {arr.dtype}"
        elif path.suffix == ".npz":
            with np.load(path) as z:
                detail = " ".join(f"{k}{z[k].shape}:{z[k].dtype}" for k in z.files)
        rows.append((path.name, size, detail))
    return {"rows": rows, "total": total}


def main() -> None:
    root = project_root()
    assets = root / "artifacts" / "app"
    if not assets.is_dir():
        raise FileNotFoundError(
            f"No assets at {assets}. Run `python scripts/build_app_assets.py` first."
        )

    meta = json.loads((assets / "meta.json").read_text())
    n_items = int(meta["n_items"])
    n_works = int(meta["n_works"])
    anchor_floor = int(meta["anchor_min_support"])

    print("=" * 78)
    print("MEASURED — the shipped demo assets")
    print("=" * 78)
    measured = measure_assets(assets)
    for name, size, detail in measured["rows"]:
        print(f"  {name:<24} {human(size):>12}   {detail}")
    print(f"  {'TOTAL':<24} {human(measured['total']):>12}")

    by_name = {name: size for name, size, _ in measured["rows"]}
    factors = by_name["factors.npy"]
    item_ids = by_name["item_ids.npy"]
    lookup = by_name["lookup_vectors.npy"] + by_name["lookup_ids.npy"] + by_name["lookup_support.npy"]

    # MEASURED: how much of the id column is padding. `<U270` pays 270 4-byte code points
    # for every id, whatever the id's length.
    ids = np.load(assets / "item_ids.npy", allow_pickle=True)
    id_bytes = sum(len(str(x).encode("utf-8")) for x in ids)
    codes = n_items * 4  # int32 code per row

    support = np.load(assets / "item_support.npy")
    askable = int((support >= anchor_floor).sum())

    print()
    print("=" * 78)
    print("MEASURED — where the 890 MB actually goes")
    print("=" * 78)
    print(f"  free-text lookup (encoder vectors + ids + support) {human(lookup):>12}"
          f"   {lookup / measured['total']:.1%}")
    print(f"  item id column, stored as fixed-width <U270        {human(item_ids):>12}"
          f"   {item_ids / measured['total']:.1%}")
    print(f"  the recommender itself (ALS factors)               {human(factors):>12}"
          f"   {factors / measured['total']:.1%}")
    print(f"  ...same ids as int32 codes + a utf-8 dictionary    {human(codes + id_bytes):>12}"
          f"   ({human(codes)} codes + {human(id_bytes)} text,"
          f" {1 - (codes + id_bytes) / item_ids:.1%} smaller)")
    print(f"  askable anchors (support >= {anchor_floor})                        {askable:>12,}"
          f"   of {n_works:,} works")

    print()
    print("=" * 78)
    print("DERIVED — what a production serving tier would hold")
    print("=" * 78)
    topn_rows = askable * TOP_N
    topn_bytes = topn_rows * 12  # int32 anchor, int32 item, float32 score
    print(f"  precomputed top-{TOP_N}, one engine    {topn_rows:>12,} rows  {human(topn_bytes):>12}")
    print(f"  precomputed top-{TOP_N}, {ENGINES} engines   {topn_rows * ENGINES:>12,} rows"
          f"  {human(topn_bytes * ENGINES):>12}")
    nn = n_items * ITEM_ITEM_NEIGHBOURS
    print(f"  item-item, {ITEM_ITEM_NEIGHBOURS} neighbours/item {nn:>12,} entries"
          f" {human(nn * 8):>12}")
    print(f"  ALS factors, {ALS_FACTORS} dims float32 {n_items * ALS_FACTORS:>12,} cells"
          f" {human(n_items * ALS_FACTORS * 4):>12}")
    dense = n_items * n_items * 4
    print(f"  the dense similarity matrix nobody builds {n_items ** 2:>12,} cells"
          f" {dense / 1e9:>9,.0f} GB")
    print()
    print(f"  ratio: shipped assets / precomputed top-{TOP_N} table ({ENGINES} engines)"
          f" = {measured['total'] / (topn_bytes * ENGINES):,.0f}x")


if __name__ == "__main__":
    main()
