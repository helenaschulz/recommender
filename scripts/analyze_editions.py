"""Measure the edition clustering: how much of the catalogue is duplicated, and what it costs.

    python scripts/analyze_editions.py
    python scripts/analyze_editions.py --write-sample docs/edition_clusters_sample.md

Prints the M11.2 statistics (works vs ISBNs, interactions per work, the sub-threshold
editions the standard min-5 filter throws away, and the corrected count against ledger
L15) and renders the M11.3 validation sample: 30 seeded-random multi-ISBN clusters, laid
out so a human can count the wrong merges. No model is fitted here — this is a property
of the catalogue, measured before anything is trained on it.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from recommender.data import cluster_works, load, to_work_level
from recommender.split import make_split

SAMPLE_SIZE = 30
SAMPLE_SEED = 42


def cluster_statistics(catalog, works) -> None:
    """M11.2: the numbers that go into the ledger."""
    books = catalog.books
    ratings = catalog.ratings
    n_isbn = len(books)
    n_works = works.n_works

    per_work = works.work_of_isbn.value_counts()
    multi = per_work[per_work > 1]

    print("=" * 88)
    print("1 · The catalogue at work level")
    print("=" * 88)
    print(f"catalogue ISBNs                     {n_isbn:>9,}")
    print(f"distinct works                      {n_works:>9,}  ({n_works / n_isbn:.1%} of the ISBN count)")
    print(f"works with more than one ISBN       {len(multi):>9,}")
    print(f"  ISBNs they cover                  {int(multi.sum()):>9,}  ({multi.sum() / n_isbn:.1%} of the catalogue)")
    print(f"  ISBNs removed by collapsing them  {int(multi.sum()) - len(multi):>9,}")
    print(f"largest cluster                     {int(per_work.max()):>9,} ISBNs  ({per_work.idxmax()})")
    print(f"books carrying a series parenthetical {(books['series'] != '').sum():>7,}")
    print("  most common series values:")
    for value, count in books.loc[books["series"] != "", "series"].value_counts().head(8).items():
        print(f"    {count:>5,}  {value}")

    # --- L15 was a lower bound: the same count under the exact title+author key. ---
    exact = (
        books["Book-Title"].fillna("").str.strip().str.lower()
        + "|"
        + books["Book-Author"].fillna("").str.strip().str.lower()
    )
    exact_sizes = books.assign(_k=exact).groupby("_k")["ISBN"].nunique()
    exact_multi = exact_sizes[exact_sizes > 1]
    pinned = cluster_works(books, merge_author_variants=False)
    pinned_sizes = pinned.work_of_isbn.value_counts()
    pinned_multi = pinned_sizes[pinned_sizes > 1]

    print()
    print("=" * 88)
    print("2 · Against ledger L15 (which was a lower bound)")
    print("=" * 88)
    print(f"{'key':<52} {'works':>8} {'ISBNs':>9}")
    print(f"{'L15: exact lower-cased title + full author':<52} {len(exact_multi):>8,} {int(exact_multi.sum()):>9,}")
    print(
        f"{'M11 pinned: normalized title + author surname':<52} "
        f"{len(pinned_multi):>8,} {int(pinned_multi.sum()):>9,}"
    )
    print(f"{'M11 shipped: + transliteration variants':<52} {len(multi):>8,} {int(multi.sum()):>9,}")
    extra = int(multi.sum()) - int(exact_multi.sum())
    changed = works.work_of_isbn.to_numpy() != pinned.work_of_isbn.to_numpy()
    n_changed_clusters = len(set(works.work_of_isbn.to_numpy()[changed]))
    print(f"\nL15 undercounts duplicated ISBNs by {extra:,} ({extra / int(exact_multi.sum()):.0%} more than reported).")
    print(
        f"The transliteration pass changes {n_changed_clusters:,} clusters, worth "
        f"{int(multi.sum()) - int(pinned_multi.sum()):,} duplicated ISBNs. The multi-ISBN cluster *count* rises by "
        f"only {len(multi) - len(pinned_multi):,}, because merging two singletons adds a cluster while merging a "
        f"singleton into an existing cluster does not."
    )

    # --- Interactions per work vs per ISBN. ---
    work_of_rating = works.of(ratings["ISBN"])
    per_isbn_inter = ratings.groupby("ISBN").size()
    per_work_inter = pd.Series(work_of_rating).value_counts()

    print()
    print("=" * 88)
    print("3 · Interactions per work vs per ISBN (all 1,149,780 interactions)")
    print("=" * 88)
    bands = [(1, 1), (2, 4), (5, 19), (20, 49), (50, 10**9)]
    print(f"{'interactions':<16} {'ISBNs':>12} {'share':>8}   {'works':>12} {'share':>8}")
    for lo, hi in bands:
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**8 else f"{lo}-{hi}")
        i_n = int(((per_isbn_inter >= lo) & (per_isbn_inter <= hi)).sum())
        w_n = int(((per_work_inter >= lo) & (per_work_inter <= hi)).sum())
        print(
            f"{label:<16} {i_n:>12,} {i_n / len(per_isbn_inter):>7.1%}   "
            f"{w_n:>12,} {w_n / len(per_work_inter):>7.1%}"
        )
    print(f"{'total':<16} {len(per_isbn_inter):>12,} {'':>8}   {len(per_work_inter):>12,}")
    print(f"\nmedian interactions per rated ISBN {per_isbn_inter.median():.0f}, per rated work "
          f"{per_work_inter.median():.0f}; mean {per_isbn_inter.mean():.2f} -> {per_work_inter.mean():.2f}")

    # --- What the standard min-5 filter throws away. ---
    print()
    print("=" * 88)
    print("4 · Sub-threshold editions of works that clear the min-5 filter")
    print("=" * 88)
    isbn_work = pd.DataFrame({"ISBN": per_isbn_inter.index, "n": per_isbn_inter.to_numpy()})
    isbn_work["work"] = works.of(isbn_work["ISBN"])
    work_total = isbn_work.groupby("work")["n"].transform("sum")
    dropped = isbn_work["n"] < 5
    rescued = dropped & (work_total >= 5)
    print(f"rated ISBNs                                    {len(isbn_work):>9,}")
    print(f"  below the min-5 threshold                    {int(dropped.sum()):>9,}  "
          f"({int(isbn_work.loc[dropped, 'n'].sum()):,} interactions)")
    print(f"  ...but part of a work that clears it          {int(rescued.sum()):>9,}  "
          f"({int(isbn_work.loc[rescued, 'n'].sum()):,} interactions)")
    surviving_isbn = int((isbn_work["n"] >= 5).sum())
    surviving_work = int((per_work_inter >= 5).sum())
    print(f"items surviving min-5: {surviving_isbn:,} ISBNs -> {surviving_work:,} works "
          f"({surviving_work / surviving_isbn - 1:+.1%})")

    # --- The L20 ceiling, recomputed at work level. ---
    # A HitRate is only readable against the share of held-out items a co-occurrence model
    # could reach at all. Clustering moves that denominator, so the work-level accuracy
    # number in L43 must be read against the work-level ceiling, not against L20's 84.8%.
    print()
    print("=" * 88)
    print("5 · The structural ceiling (L20) at both levels")
    print("=" * 88)
    isbn_split = None
    for label, frame in (("ISBN level", ratings), ("work level", to_work_level(ratings, works))):
        split = make_split(frame)
        isbn_split = isbn_split or split
        train_items = set(split.train["ISBN"].unique())
        reachable = split.test["ISBN"].isin(train_items)
        print(
            f"{label:<12} {split.n_eligible:>7,} eligible users   ceiling "
            f"{reachable.mean():.2%}  ({int(reachable.sum()):,} of {len(reachable):,} held-out items in train)"
        )

    # Could the ISBN-level number have been flattered by its own duplicates? A held-out
    # ISBN whose *work* the user also owns under a second ISBN is a near-free hit for any
    # co-occurrence model, and work level removes that path. Worth ruling out before
    # reading the work-level lift as real.
    train_pairs = set(zip(isbn_split.train["User-ID"], works.of(isbn_split.train["ISBN"]), strict=True))
    same_work = sum(
        (user, work) in train_pairs
        for user, work in zip(isbn_split.test["User-ID"], works.of(isbn_split.test["ISBN"]), strict=True)
    )
    print(
        f"\nISBN-level holdouts whose work the user already owns under another ISBN: "
        f"{same_work:,} of {isbn_split.n_eligible:,} ({same_work / isbn_split.n_eligible:.2%}) "
        f"— too few to explain any work-level lift"
    )

    # Helena's motivating example, recomputed.
    cp = "crime and punishment|dostoevsky"
    members = works.work_of_isbn[works.work_of_isbn == cp].index
    counts = per_isbn_inter.reindex(members).fillna(0).astype(int).sort_values(ascending=False)
    print(f"\nThe motivating case — '{cp}': {len(members)} ISBNs, {int(counts.sum())} interactions, "
          f"strongest edition {int(counts.iloc[0])}, editions below 5: {int((counts < 5).sum())}")


def validation_sample(catalog, works, *, size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> str:
    """M11.3: seeded-random multi-ISBN clusters, rendered for a human to audit."""
    books = catalog.books.set_index("ISBN")
    per_isbn_inter = catalog.ratings.groupby("ISBN").size()
    sizes = works.work_of_isbn.value_counts()
    multi = sizes[sizes > 1].index.to_numpy()

    rng = np.random.default_rng(seed)
    picked = rng.choice(multi, size=min(size, len(multi)), replace=False)
    picked = sorted(picked)

    members = works.work_of_isbn[works.work_of_isbn.isin(set(picked))]
    by_work: dict[str, list[str]] = {work: [] for work in picked}
    for isbn, work in members.items():
        by_work[work].append(isbn)

    lines = [
        f"# Edition-clustering validation sample ({size} clusters, seed {seed})",
        "",
        "Milestone M11.3. Every multi-ISBN cluster the work key produced had an equal chance",
        "of appearing here; the draw is seeded so the table is reproducible. The question each",
        "row answers is narrow: **are these ISBNs the same work?** A cluster is a *wrong merge*",
        "if any member is a different book, not merely a different edition, translation,",
        "printing or omnibus of the same text.",
        "",
        "This file is regenerated evidence, not a conclusion: the audit verdict for both",
        "samples is recorded in `RESULTS.md` (L42) and read out in `model_selection.md` §10.",
        "Generated by `python scripts/analyze_editions.py --write-sample`.",
        "",
    ]
    for n, work in enumerate(picked, start=1):
        isbns = sorted(by_work[work], key=lambda i: -int(per_isbn_inter.get(i, 0)))
        lines.append(f"## {n}. `{work}` — {len(isbns)} ISBNs")
        lines.append("")
        lines.append("| ISBN | Title | Author | Year | Publisher | interactions |")
        lines.append("|---|---|---|---|---|---:|")
        for isbn in isbns:
            row = books.loc[isbn]
            title = str(row["Book-Title"]).replace("|", "\\|")
            author = str(row["Book-Author"]).replace("|", "\\|")
            publisher = str(row["Publisher"]).replace("|", "\\|")
            lines.append(
                f"| {isbn} | {title} | {author} | {row['Year-Of-Publication']} | "
                f"{publisher} | {int(per_isbn_inter.get(isbn, 0))} |"
            )
        lines.append("")
    return "\n".join(lines)


def variant_sample(catalog, works, *, size: int = 20, seed: int = SAMPLE_SEED) -> str:
    """The clusters the transliteration pass created, sampled separately.

    A uniform draw over 24,392 clusters will essentially never contain one of the ~120
    the extension is responsible for, so the random sample above cannot audit the part of
    the key that deviates from the pinned decision. This one only draws from those.
    """
    pinned = cluster_works(catalog.books, merge_author_variants=False)
    changed = works.work_of_isbn.index[works.work_of_isbn.to_numpy() != pinned.work_of_isbn.to_numpy()]
    affected = sorted(set(works.work_of_isbn.loc[changed]))

    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(np.array(affected, dtype=object), size=min(size, len(affected)), replace=False))
    books = catalog.books.set_index("ISBN")
    per_isbn_inter = catalog.ratings.groupby("ISBN").size()

    lines = [
        "",
        f"# Transliteration-merge audit ({len(picked)} of {len(affected)} affected clusters, seed {seed})",
        "",
        "The random sample above draws from all 24,392 multi-ISBN clusters, so it is almost",
        "certain to contain none of the clusters the *author-variant* pass is responsible for —",
        "the extension to the pinned key (ledger L41). Those are audited here instead. A cluster",
        "is wrong if the two surname spellings belong to two different people.",
        "",
    ]
    for n, work in enumerate(picked, start=1):
        members = works.work_of_isbn[works.work_of_isbn == work].index
        isbns = sorted(members, key=lambda i: -int(per_isbn_inter.get(i, 0)))
        lines.append(f"## V{n}. `{work}` — {len(isbns)} ISBNs")
        lines.append("")
        lines.append("| ISBN | Title | Author | Year | interactions |")
        lines.append("|---|---|---|---|---:|")
        for isbn in isbns:
            row = books.loc[isbn]
            title = str(row["Book-Title"]).replace("|", "\\|")
            author = str(row["Book-Author"]).replace("|", "\\|")
            lines.append(
                f"| {isbn} | {title} | {author} | {row['Year-Of-Publication']} | {int(per_isbn_inter.get(isbn, 0))} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-sample", metavar="PATH", default=None, help="render the M11.3 table to a file")
    parser.add_argument("--no-stats", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    catalog = load()
    print(f"catalogue loaded in {time.perf_counter() - started:.0f}s", flush=True)
    t0 = time.perf_counter()
    works = catalog.works
    print(f"clustered {len(catalog.books):,} ISBNs into {works.n_works:,} works in {time.perf_counter() - t0:.1f}s\n")

    if not args.no_stats:
        cluster_statistics(catalog, works)

    if args.write_sample:
        text = validation_sample(catalog, works) + variant_sample(catalog, works)
        with open(args.write_sample, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"\nvalidation sample written to {args.write_sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
