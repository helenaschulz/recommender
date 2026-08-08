"""Price the M14.4 work-key change in isolation, before anything is re-based on it.

    python scripts/analyze_work_key_punctuation.py
    python scripts/analyze_work_key_punctuation.py --write-sample docs/work_key_punctuation_sample.md

The defect, visible on screen: *Bridget Jones's Diary* answers with *Bridget Jones: The
Edge of Reason* at rank 1 **and** *Bridget Jones : The Edge of Reason* at rank 2. They are
one book typed by two cataloguers, and the M11 key keeps them apart because it collapses
runs of whitespace without noticing that a space in front of a colon is not a word
boundary. ``WorkDeduped`` cannot help: at work level the collapse is supposed to have
happened already.

**This script changes nothing.** It runs the M11 key and the M11 key + punctuation
normalization over the same ``Books.csv`` and reports the difference, because the work key
is the basis of the published M12 table: merging more ISBNs moves ``n_works``, and
``n_works`` is the denominator of every Coverage@10 cell in it. That is the M12 procedure —
measure the key change alone, then let Helena decide whether the table is re-based or the
fix stays on the serving path.

What it prints:

1. how many works merge, how many ISBNs move, and what that is as a share;
2. the same thing weighted by **interactions**, which is the honest denominator: a merge
   nobody has read changes no recommendation;
3. the effect on the demo's own item universe, i.e. how many merges involve two works that
   both clear the app's support floor — those are the ones a visitor can actually see;
4. a seeded random sample of merged pairs, for a human to count wrong merges in, exactly
   as ``docs/edition_clusters_sample.md`` did for M11.3.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data import build_interactions, cluster_works, load, to_work_level

SAMPLE_SIZE = 30
SAMPLE_SEED = 42


def merged_groups(before: pd.Series, after: pd.Series) -> dict[str, list[str]]:
    """New work id -> the old work ids it absorbed, for every id that absorbed more than one.

    Keyed on the *new* id so a three-way merge is reported as one group rather than three
    pairs. Old ids that simply got renamed (every member moved together) are not merges and
    do not appear.
    """
    groups: dict[str, set[str]] = {}
    for old, new in zip(before.to_numpy(), after.to_numpy(), strict=True):
        groups.setdefault(new, set()).add(old)
    return {new: sorted(olds) for new, olds in groups.items() if len(olds) > 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-sample", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--app-support-floor", type=int, default=20, help="ledger L34")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    catalog = load()
    books = catalog.books
    before = cluster_works(books)
    after = cluster_works(books, normalize_punctuation=True)
    print(f"catalogue loaded and clustered twice in {time.perf_counter() - started:.0f}s\n", flush=True)

    groups = merged_groups(before.work_of_isbn, after.work_of_isbn)
    merged_old_ids = {old for olds in groups.values() for old in olds}
    renamed = int((before.work_of_isbn.to_numpy() != after.work_of_isbn.to_numpy()).sum())
    # A renamed id is not a merge. Only the ISBNs inside a merge group actually change
    # which *other* books they are pooled with, and only those can move a recommendation.
    joined = int(before.work_of_isbn.isin(merged_old_ids).sum())

    print("=" * 96)
    print("1 · What the punctuation rule merges")
    print("=" * 96)
    print(f"{'works, M11 key':<46} {before.n_works:>9,}")
    print(f"{'works, M11 key + punctuation':<46} {after.n_works:>9,}")
    print(
        f"{'works removed':<46} {before.n_works - after.n_works:>9,}  "
        f"({(before.n_works - after.n_works) / before.n_works:.3%} of the work universe)"
    )
    print(f"{'merge groups':<46} {len(groups):>9,}")
    print(f"{'ISBNs in a merge group':<46} {joined:>9,}  ({joined / len(books):.3%} of the catalogue)")
    print(
        f"{'ISBNs whose work id string changes at all':<46} {renamed:>9,}  "
        f"({renamed / len(books):.3%}) — the rest of these are renames, not merges"
    )

    # --- 2 · The same thing weighted by interactions --------------------------------
    work_before = to_work_level(catalog.ratings, before)
    work_after = to_work_level(catalog.ratings, after)
    per_old = catalog.ratings.assign(_w=before.of(catalog.ratings["ISBN"])).groupby("_w").size()
    touched = int(per_old.reindex(sorted(merged_old_ids)).fillna(0).sum())

    print()
    print("=" * 96)
    print("2 · Weighted by interactions, which is what a reader would notice")
    print("=" * 96)
    print(f"{'interactions on a merged work':<46} {touched:>9,}  ({touched / len(catalog.ratings):.3%} of all)")
    print(f"{'(user, work) rows, M11 key':<46} {len(work_before):>9,}")
    print(f"{'(user, work) rows, + punctuation':<46} {len(work_after):>9,}")
    print(
        f"{'rows collapsed (one reader, two editions)':<46} {len(work_before) - len(work_after):>9,}  "
        f"({(len(work_before) - len(work_after)) / len(work_before):.4%})"
    )

    # --- 3 · How much of it the demo can actually show ------------------------------
    interactions = build_interactions(work_before, weights="binary")
    support = dict(zip(interactions.item_ids.tolist(), interactions.item_popularity.tolist(), strict=True))
    floor = args.app_support_floor
    visible = {
        new: olds for new, olds in groups.items() if sum(support.get(old, 0) >= floor for old in olds) >= 2
    }
    print()
    print("=" * 96)
    print(f"3 · Merges the demo can show: both sides above the support floor of {floor}")
    print("=" * 96)
    print(f"{'merge groups with >=2 answerable members':<46} {len(visible):>9,}")
    print(f"{'... as a share of all merge groups':<46} {len(visible) / max(len(groups), 1):>9.1%}")
    print("\nthe ten with the most interactions on the smaller side:")
    ranked = sorted(
        visible.items(),
        key=lambda item: -min(support.get(old, 0) for old in item[1]),
    )[:10]
    for new, olds in ranked:
        sizes = ", ".join(f"{old!r} ({support.get(old, 0)})" for old in sorted(olds, key=lambda o: -support.get(o, 0)))
        print(f"  -> {new!r}\n       {sizes}")

    # --- 4 · The audit sample -------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    keys = sorted(groups)
    picked = [keys[i] for i in rng.choice(len(keys), size=min(args.sample_size, len(keys)), replace=False)]
    label = before.label_of_work
    lines = [
        "# M14.4 · The punctuation merge, sampled for a human to count wrong merges in",
        "",
        f"{len(groups):,} merge groups; {len(picked)} drawn with `numpy.default_rng({args.seed})` by "
        f"`scripts/analyze_work_key_punctuation.py`. Each row is one group: the work ids the M11 key "
        "produced, the id they merge into, and a title from each side so the pair can be judged without "
        "the data.",
        "",
        "| # | merges into | old work ids | titles |",
        "|--:|---|---|---|",
    ]
    for i, new in enumerate(picked, start=1):
        olds = groups[new]
        titles = " <br> ".join(str(label.get(old, "[no metadata]")) for old in olds)
        lines.append(f"| {i} | `{new}` | " + "<br>".join(f"`{old}`" for old in olds) + f" | {titles} |")
    sample = "\n".join(lines) + "\n"

    print()
    print("=" * 96)
    print(f"4 · Audit sample ({len(picked)} groups, seed {args.seed})")
    print("=" * 96)
    print(sample)
    if args.write_sample:
        args.write_sample.write_text(sample)
        print(f"written to {args.write_sample}")

    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
