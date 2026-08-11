"""The twenty anchors under both configurations, as a table a human reads (M23.10.3).

    python scripts/audit_anchor_set.py                    # docs/anchor_set_audit.md
    python scripts/audit_anchor_set.py --k 10

**This is a gate, not a report.** M23.10 decision 3: L89 audited the **20-49** reader band and
every rehearsed anchor is **50+**, so configuration B's lists up there are new and have never
been read by anyone. A *Harry Potter* that does not return its sequels under B is discoverable
on stage. So B's top-10 for all twenty anchors is written out beside A's for the same anchor,
a person reads both columns, and the picker does not ship unless the reading passes. This
script produces the table; it does not grade it — the same rule as
``scripts/audit_floor_band.py``, for the same reason (L42, L79, L89 were all hand-counted, and
a script that graded its own output would be measuring a title-matching rule).

**Anchors are resolved through the app's own** :meth:`~recommender.demo.DemoEngine.find`
**path, from the title a visitor would type**, not by work id. That is deliberate: the gate is
supposed to test what somebody actually gets when they type *The Curious Incident of the Dog in
the Night-Time*, including the case where the work key has split the book in two and the query
lands on the larger half. Addressing anchors by id would step over exactly the defect the set
was chosen to surface.

**It also prints the numbers M23.10.7 needs, so they stop being numbers from a session.** The
milestone's anchor table carries reader counts read off the shipped assets by hand; this
script prints them from the assets, with the split-work analysis beside them, so the ledger
line can cite a command. That is the M16 correction's standing rule: numbers without a command
do not enter this ledger.

Three sections come out of it:

1. **Resolution** — what each typed title resolves to, its reader count, and what else the
   picker would have offered.
2. **The two lists** — A and B side by side per anchor, with the co-reader count the app
   prints under each slot.
3. **Work-key splits** — the two open classes named with instances (leading article, subtitle),
   plus how often each class costs a work its place above the anchor floor.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from recommender.answers import askable_rows
from recommender.data import split_series
from recommender.demo import DemoEngine, load_assets
from recommender.models.embeddings import sentence_transformer_encoder

OUT = Path("docs/anchor_set_audit.md")

#: The twenty, exactly as the M23.10 table lists them and in its order — the rehearsed
#: eleven first, then the nine added on 10.08. The key is written as a **visitor would type
#: it**, because that is what the gate is testing; the value is the work the milestone's table
#: means by it.
#:
#: **The two are checked against each other rather than assumed equal**, and the check earns
#: its keep: two of the twenty do not resolve to the book they name. Where they differ, both
#: anchors are read — the one a visitor lands on, because that is what the gate is for, and the
#: one the set intended, because that is the book the set was chosen to be able to judge.
ANCHOR_QUERIES: dict[str, str] = {
    "The Lovely Bones": "the lovely bones: a novel|sebold",
    "The Da Vinci Code": "the da vinci code|brown",
    "Harry Potter and the Sorcerer's Stone": "harry potter and the sorcerer's stone|rowling",
    "Bridget Jones's Diary": "bridget jones's diary|fielding",
    "Girl with a Pearl Earring": "girl with a pearl earring|chevalier",
    "Interview with the Vampire": "interview with the vampire|rice",
    "To Kill a Mockingbird": "to kill a mockingbird|lee",
    "The Hobbit": "the hobbit: the enchanting prelude to the lord of the rings|tolkien",
    "Dune": "dune|herbert",
    "Fight Club": "fight club|palahniuk",
    "Guns, Germs, and Steel": "guns, germs, and steel: the fates of human societies|diamond",
    "Life of Pi": "life of pi|martel",
    "Tuesdays with Morrie": "tuesdays with morrie: an old man, a young man, and life's greatest lesson|albom",
    "Angela's Ashes": "angela's ashes (mmp): a memoir|mccourt",
    "Love in the Time of Cholera": "love in the time of cholera|marquez",
    "One Hundred Years of Solitude": "one hundred years of solitude|marquez",
    "The Curious Incident of the Dog in the Night-Time": "the curious incident of the dog in the night-time|haddon",
    "Crime and Punishment": "crime and punishment|dostoevsky",
    "War and Peace": "war and peace|tolstoy",
    "The Master and Margarita": "the master and margarita|bulgakov",
}

#: Queried and reported, never part of the twenty. *The Kite Runner* is decision 5's floor
#: demonstration — answerable by neither configuration, because at a common floor the refusal
#: is a property of the floor rather than of the engine. *The Purpose Driven Life* is the
#: representativeness question M23.10.7 asks. *The Brothers Karamazov* is the leading-article
#: split, both halves under the floor.
OFF_SET_QUERIES: tuple[str, ...] = (
    "The Kite Runner",
    "The Purpose Driven Life",
    "The Brothers Karamazov",
)

#: Work keys M23.10.7 has to print a number for, whether or not a query reaches them. Named
#: here so the ledger line cites a command rather than a session, which is the M16 rule.
LEDGER_WORKS: tuple[str, ...] = (
    "the kite runner|hosseini",
    "the purpose-driven life: what on earth am i here for?|warren",
    "the purpose driven life: what on earth am i here for?|warren",
    "the brothers karamazov|dostoevsky",
    "brothers karamazov|dostoevsky",
    "the curious incident of the dog in the night-time|haddon",
    "the curious incident of the dog in the night-time: a novel|haddon",
)

#: The two configurations M23.10 ships, in picker order.
COLUMNS: tuple[str, ...] = ("A", "B")

_ARTICLE = re.compile(r"^(the|a|an)\s+")
_PUNCTUATION = re.compile(r"[^\w\s]+")
#: A *mid-string* parenthetical. The **trailing** one is already gone: the work key is built
#: through ``split_series``, so ``dune (remembering tomorrow)`` never reaches this module as an
#: id. What survives is the kind sitting in the middle of a title — ``angela's ashes (mmp): a
#: memoir`` — which is why that book is split three ways and none of the other rules can see it.
_PARENTHETICAL = re.compile(r"\([^)]*\)")


def work_parts(work_id: str) -> tuple[str, str]:
    """``title|author`` split back into its two halves; the author half may contain no bar."""
    title, _, author = work_id.rpartition("|")
    return (title, author) if title else (work_id, "")


def collapsed_key(work_id: str) -> str:
    """The work key with the four known defects normalized away, for grouping only.

    Drops a mid-string parenthetical, everything after the first colon, a leading article, and
    internal punctuation. **This is a diagnostic, not a proposal**: it is what makes *the
    brothers karamazov|dostoevsky* and *brothers karamazov|dostoevsky* land in one bucket so
    the split can be counted. Shipping it as the work key is a data-layer change that would
    move every published number — L67 measured that a re-key touching 0.5% of works replaced a
    third of every neighbourhood — which is the whole reason these classes are still open
    items rather than fixes.
    """
    title, author = work_parts(work_id)
    title = _PARENTHETICAL.sub(" ", title)
    title = title.split(":")[0].strip()
    title = _ARTICLE.sub("", title).strip()
    return f"{' '.join(_PUNCTUATION.sub(' ', title).split())}|{author}"


def split_class(a: str, b: str) -> str:
    """Which of the four open defect classes separates these two work ids, if any.

    Checked in the order a reader would notice them, and each is a class this ledger already
    has an open item for: a subtitle one half carries and the other does not; a mid-string
    parenthetical, L48's field one layer up; a leading article the key does not strip (L89,
    found on *Songlines*); and internal punctuation that M14.4's normalization does not reach
    (L64 normalized between the title and the author, not inside the title). ``other`` is the
    residue — two halves whose titles differ in words, e.g. *Mount* against *Mt.* — and it is
    reported rather than folded into the four, because it is not one defect.
    """
    title_a, _ = work_parts(a)
    title_b, _ = work_parts(b)
    if (":" in title_a) != (":" in title_b):
        return "subtitle"
    if bool(_PARENTHETICAL.search(title_a)) != bool(_PARENTHETICAL.search(title_b)):
        return "parenthetical"
    if bool(_ARTICLE.match(title_a)) != bool(_ARTICLE.match(title_b)):
        return "leading article"
    if _PUNCTUATION.sub(" ", title_a).split() != _PUNCTUATION.sub(" ", title_b).split():
        return "other"
    return "internal punctuation"


def sibling_groups(assets) -> dict[str, list[int]]:
    """Every :func:`collapsed_key` bucket holding more than one **nameable** work row."""
    named = set(assets.books.index)
    groups: dict[str, list[int]] = defaultdict(list)
    for row, work_id in enumerate(assets.item_ids.tolist()):
        if work_id in named:
            groups[collapsed_key(work_id)].append(row)
    return {key: rows for key, rows in groups.items() if len(rows) > 1}


def resolve_row(engine: DemoEngine, index: dict[str, int], query: str) -> dict[str, object]:
    """One typed query through the app's own input path, with everything it offered."""
    resolution = engine.resolve(query, k=5)
    matches = resolution.matches
    chosen = matches[0] if matches else None
    return {
        "query": query,
        "book": chosen,
        "alternatives": matches[1:],
        "below_floor": resolution.below_floor,
        "row": index.get(chosen.isbn) if chosen else None,
    }


#: Accumulated by :func:`render_lists`, printed as section 6. **The script counts what is
#: countable and the human counts what is judgeable** — a slot's co-reader count is a number,
#: so a script may add it up; whether a slot is a *comparable read* is not, which is why
#: section 5 stays empty until a person fills it in. Same split as
#: ``scripts/audit_floor_band.py``, and the same reason (L42, L79, L89).
TALLY: dict[str, list[int]] = defaultdict(list)


def render_lists(engines: dict[str, DemoEngine], work_id: str, k: int) -> list[str]:
    """A and B side by side for one anchor, in the shape a reader can scan a row at a time."""
    answers = {label: engine.similar(work_id, k=k) for label, engine in engines.items()}
    for label, got in answers.items():
        TALLY[label] += [item.evidence.co_readers for item in got]
    header = " | ".join(f"{label} · {engines[label].configuration.short_label} | co" for label in COLUMNS)
    lines = [
        "| # | " + header + " |",
        "|---|" + "---|---:|" * len(COLUMNS),
    ]
    for rank in range(k):
        cells: list[str] = []
        for label in COLUMNS:
            got = answers[label]
            if rank < len(got):
                item = got[rank]
                title = split_series(item.title)[0].replace("|", "\\|")
                cells += [f"{title} by *{item.author}*", f"{item.evidence.co_readers:,}"]
            else:
                cells += ["—", "—"]
        lines.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    assets = load_assets()
    encoder = sentence_transformer_encoder(assets.encoder_model)
    engines = {label: DemoEngine(assets, encoder=encoder, configuration=label) for label in COLUMNS}
    index = assets.item_index
    support = assets.item_support
    print(f"engines built in {time.perf_counter() - started:.0f}s", flush=True)

    resolved = [resolve_row(engines["A"], index, query) for query in ANCHOR_QUERIES]
    for row in resolved:
        intended = ANCHOR_QUERIES[str(row["query"])]
        book = row["book"]
        row["intended"] = intended
        row["landed"] = book is not None and book.isbn == intended
    off_set = [resolve_row(engines["A"], index, query) for query in OFF_SET_QUERIES]
    strays = [row for row in resolved if not row["landed"]]
    print(
        f"resolved {len(resolved) + len(off_set)} queries, "
        f"{len(strays)} did not land on the intended work [{time.perf_counter() - started:.0f}s]",
        flush=True,
    )

    groups = sibling_groups(assets)

    lines = [
        "# The twenty anchors under both configurations (milestone M23.10.3)",
        "",
        f"`python scripts/audit_anchor_set.py` — {date.today().isoformat()}. "
        f"{len(ANCHOR_QUERIES)} anchors × {len(COLUMNS)} configurations × {args.k} slots = "
        f"**{len(ANCHOR_QUERIES) * len(COLUMNS) * args.k} slots** to read.",
        "",
        "Configurations: "
        + " · ".join(f"**{label}** = {engines[label].configuration.short_label}" for label in COLUMNS)
        + f". **Both sit at anchor floor {assets.anchor_floor}** and the candidate floor stays "
        f"{assets.similar_min_support} in both (L34), so the switch moves one variable.",
        "",
        "`co` is the number of the anchor's readers who also read that book — the number the "
        "app prints. The `score` column is deliberately **not** here: A's is a cosine between "
        "learned profiles and B's a shrunk cosine over shared readers, they are not comparable, "
        "and a table that put them side by side would invite exactly the comparison M23 "
        "decision 6 exists to prevent.",
        "",
        "Every anchor below was reached by typing the title into the app's own search box, not "
        "by work id — see the resolution table for what each query actually landed on.",
        "",
        "## 1 · What each typed title resolves to",
        "",
        f"**{len(strays)} of {len(resolved)} do not land on the book the set names.** Both are "
        "read in section 3 where they differ: the book a visitor gets, because that is what "
        "the gate is for, and the book the set intended, because that is the one chosen to be "
        "judgeable.",
        "",
        "| # | typed | resolved to | readers | ✓ | also offered |",
        "|--:|---|---|--:|:-:|---|",
    ]
    for number, (query, row) in enumerate(zip(ANCHOR_QUERIES, resolved, strict=True), start=1):
        book = row["book"]
        if book is None:
            lines.append(f"| {number} | {query} | **nothing** | | ✗ | |")
            continue
        alternatives = ", ".join(f"{b.title} ({b.readers:,})" for b in row["alternatives"]) or "—"
        lines.append(
            f"| {number} | {query} | {split_series(book.title)[0]} by *{book.author}* | "
            f"{book.readers:,} | {'✓' if row['landed'] else '**✗**'} | {alternatives} |"
        )
    lines.append("")

    lines += [
        "## 2 · Queried but not in the set",
        "",
        "`resolves to` is what the app answers. `the floor is hiding` is the group M23.10.5 "
        "adds to the picker: books the query matches **better** than anything the demo can "
        "answer, which the search box has dropped silently since M13.",
        "",
        "| typed | resolves to | readers | the floor is hiding |",
        "|---|---|--:|---|",
    ]
    for row in off_set:
        book = row["book"]
        hidden = row["below_floor"]
        landed = f"{split_series(book.title)[0]} by *{book.author}*" if book else "**nothing**"
        landed_readers = f"{book.readers:,}" if book else "—"
        hidden_text = (
            " · ".join(f"{split_series(b.title)[0]} by *{b.author}* ({b.readers:,})" for b in hidden)
            or "*nothing: the floor removes nothing this query ranks higher*"
        )
        lines.append(f"| {row['query']} | {landed} | {landed_readers} | {hidden_text} |")
    lines.append("")

    lines += [
        "## 2b · The work keys M23.10.7 has to cite",
        "",
        "| work id | readers | askable at the floor |",
        "|---|--:|:-:|",
    ]
    for work_id in LEDGER_WORKS:
        row_index = index.get(work_id)
        if row_index is None:
            lines.append(f"| `{work_id}` | *not in the assets* | |")
            continue
        readers = int(support[row_index])
        lines.append(f"| `{work_id}` | {readers:,} | {'yes' if readers >= assets.anchor_floor else 'no'} |")
    lines.append("")

    lines += ["## 3 · The two lists, anchor by anchor", ""]
    for number, row in enumerate(resolved, start=1):
        book = row["book"]
        query = str(row["query"])
        if book is not None:
            lines += [
                f"### {number}. {split_series(book.title)[0]} by {book.author}",
                "",
                f"typed *{query}* · `{book.isbn}` · **{book.readers:,} readers**"
                + ("" if row["landed"] else "  ·  ⚠ **not the book this query names**"),
                "",
            ]
            lines += render_lists(engines, book.isbn, args.k)
            print(f"  {number:>2}. {book.title}", flush=True)
        if row["landed"]:
            continue
        intended = str(row["intended"])
        wanted = engines["A"].describe(intended)
        lines += [
            f"### {number}b. {split_series(wanted.title)[0]} by {wanted.author}",
            "",
            f"the anchor the set means by *{query}*, reached by work id because the search box "
            f"does not reach it · `{intended}` · **{wanted.readers:,} readers**",
            "",
        ]
        lines += render_lists(engines, intended, args.k)
        print(f"  {number:>2}b. {wanted.title} (intended)", flush=True)

    # -- section 4: the work-key splits M23.10.7 has to name --------------------------------
    lines += [
        "## 4 · Work-key splits",
        "",
        "A *split* here means two nameable work ids that differ only by a subtitle, a "
        "parenthetical, a leading article or internal punctuation — two rows the catalogue "
        "treats as separate books when a reader would call them one. The classes are open "
        "items in this ledger and had no named instance until this set was read.",
        "",
        "### The anchors' own siblings",
        "",
        "Seeded from the twenty works the set names plus the keys M23.10.7 cites, so a split "
        "shows up here whether or not the search box reaches either half.",
        "",
        "| seed | readers | the other half | readers | together | class |",
        "|---|--:|---|--:|--:|---|",
    ]
    seeds = list(dict.fromkeys(list(ANCHOR_QUERIES.values()) + list(LEDGER_WORKS)))
    for seed_id in seeds:
        seed_row = index.get(seed_id)
        if seed_row is None:
            continue
        siblings = [r for r in groups.get(collapsed_key(seed_id), []) if r != seed_row]
        for sibling in sorted(siblings, key=lambda r: -support[r]):
            other = str(assets.item_ids[sibling])
            total = int(support[seed_row]) + int(support[sibling])
            lines.append(
                f"| `{seed_id}` | {int(support[seed_row]):,} | `{other}` | "
                f"{int(support[sibling]):,} | {total:,} | {split_class(seed_id, other)} |"
            )
    lines.append("")

    # How often each class costs a work its place above the floor, over the whole catalogue.
    floor = assets.anchor_floor
    tally: dict[str, list[int]] = defaultdict(list)
    silenced: dict[str, int] = defaultdict(int)
    for rows in groups.values():
        supports = sorted((int(support[r]) for r in rows), reverse=True)
        ids = sorted(rows, key=lambda r: -support[r])
        klass = split_class(str(assets.item_ids[ids[0]]), str(assets.item_ids[ids[1]]))
        tally[klass].append(len(rows))
        if supports[0] < floor <= sum(supports):
            silenced[klass] += 1
    lines += [
        "### How much each class costs, over the whole nameable catalogue",
        "",
        f"| class | split groups | works involved | groups the split silences at floor {floor} |",
        "|---|--:|--:|--:|",
    ]
    for klass in sorted(tally, key=lambda c: -len(tally[c])):
        lines.append(
            f"| {klass} | {len(tally[klass]):,} | {sum(tally[klass]):,} | {silenced.get(klass, 0):,} |"
        )
    total_silenced = sum(silenced.values())
    askable = len(askable_rows(assets))
    lines += [
        f"| **all classes** | **{sum(len(v) for v in tally.values()):,}** | "
        f"**{sum(sum(v) for v in tally.values()):,}** | **{total_silenced:,}** |",
        "",
        "*Silenced* means every half sits below the anchor floor while their sum clears it: the "
        "book exists in the data often enough to be answerable and the key hides it. That is "
        "the same defect L89 found on *Songlines*, counted rather than instanced.",
        "",
        f"**{total_silenced:,} silenced groups against {askable:,} askable works**: merging the "
        f"halves would take the askable catalogue to {askable + total_silenced:,}, "
        f"**+{total_silenced / askable:.1%}**, without touching the floor or the engine. Stated "
        "as an upper bound on this defect and not as a proposal — the merge is a data-layer "
        "change and L67 measured what a re-key does to a neighbourhood.",
        "",
        "## 5 · The count",
        "",
        "Filled in by hand after reading section 3, under **L79's criterion, inherited "
        "unchanged** so this count and L89's are comparable: a slot is bad when it is the "
        "anchor itself under another edition, title or translation; a companion or *about* "
        "book rather than a comparable read; or a duplicate of another slot in the same list. "
        "Empty until someone has read it — a count written by the script that produced the "
        "lists is the thing this audit exists to avoid.",
        "",
        "| configuration | bad slots | of | date |",
        "|---|--:|--:|---|",
        f"| A · {engines['A'].configuration.short_label} | | {len(TALLY['A'])} | |",
        f"| B · {engines['B'].configuration.short_label} | | {len(TALLY['B'])} | |",
        "",
        "## 6 · The evidence behind the slots, counted by the script",
        "",
        "Not a judgment and therefore not the human's to make: these are L87's definitions "
        "applied to this anchor set, over every list in section 3.",
        "",
        "| configuration | slots | median co-readers | thin (<5 co-readers) | zero co-readers | weakest slot |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for label in COLUMNS:
        counts = sorted(TALLY[label])
        thin = sum(1 for c in counts if c < 5)
        zero = sum(1 for c in counts if c == 0)
        median = counts[len(counts) // 2] if counts else 0
        lines.append(
            f"| {label} · {engines[label].configuration.short_label} | {len(counts)} | {median} | "
            f"{thin} ({thin / max(len(counts), 1):.1%}) | {zero} | {counts[0] if counts else 0} |"
        )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {args.out} — {len(ANCHOR_QUERIES)} anchors, "
        f"{len(ANCHOR_QUERIES) * len(COLUMNS) * args.k} slots  "
        f"[{time.perf_counter() - started:.0f}s]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
