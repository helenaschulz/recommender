"""Audit the demo's free-text box: what each serving rule is worth, on the L37/L38 queries.

    python scripts/audit_app_lookup.py

The app's input path is the part a panel will actually poke at, and it carries two rules
that the embedding model itself does not (:mod:`recommender.demo`):

1. a **support floor** — never offer an anchor the engine would refuse to answer for;
2. a **tie margin** — among works whose titles match almost equally well, prefer the one
   most readers mean.

Both are serving rules for a demo, not model changes, and neither touches a published
metric. That is exactly why they need auditing rather than trusting: a rule that quietly
improves the queries you happened to look at is how a demo starts lying. This script scores
the same query set L37 and L38 used, under all three conditions, and prints what each rule
moved — including the two queries that **no** rule fixes, which are L38's actual finding.
"""

from __future__ import annotations

import sys

import numpy as np

from recommender.demo import LOOKUP_TIE_MARGIN, DemoEngine, load_assets

#: The L37/L38 query set, plus three unambiguous controls. "expected" is a substring the
#: rank-1 title must contain to count as resolved; None means "L38 measured this as failing".
QUERIES: list[tuple[str, str | None]] = [
    ("harry potter stein", "Sorcerer's Stone"),
    ("da vinci code", "Da Vinci Code"),
    ("lovely bones", "Lovely Bones"),
    ("el senor de los anillos", "Ring"),
    ("der kleine prinz", "Prince"),
    ("the hobbit", "Hobbit"),
    ("lord of the rings", "Ring"),
    ("angels and demons", "Angels"),
    ("life of pi", "Life of Pi"),
    ("herr der ringe", None),
    ("hobit tolkien", None),
]


def main(argv: list[str] | None = None) -> int:
    assets = load_assets()
    engine = DemoEngine(assets)
    titles = assets.books["Book-Title"].to_dict()
    support = assets.lookup_support

    def top1(query: str, *, floor: bool, margin: float) -> tuple[str, int]:
        vector = engine._encode(query)  # noqa: SLF001 - this script audits the engine
        scores = np.asarray(assets.lookup_vectors @ vector)
        if floor:
            scores = np.where(support >= assets.similar_min_support, scores, -np.inf)
        order = np.argsort(-scores)[:60]
        order = order[np.isfinite(scores[order])]
        if margin:
            cutoff = scores[order[0]] - margin
            tied = sorted((r for r in order if scores[r] >= cutoff), key=lambda r: (-support[r], -scores[r]))
            order = np.array(tied + [r for r in order if scores[r] < cutoff])
        row = order[0]
        return str(titles.get(str(assets.lookup_ids[row]), "")), int(support[row])

    conditions = (
        ("cosine only", {"floor": False, "margin": 0.0}),
        ("+ support floor", {"floor": True, "margin": 0.0}),
        ("+ tie margin", {"floor": True, "margin": LOOKUP_TIE_MARGIN}),
    )
    tally = dict.fromkeys((name for name, _ in conditions), 0)
    answerable = sum(1 for _, expected in QUERIES if expected)

    print(f"{'query':<26}" + "".join(f"{name:<44}" for name, _ in conditions))
    for query, expected in QUERIES:
        cells = []
        for name, kwargs in conditions:
            title, readers = top1(query, **kwargs)
            hit = expected is not None and expected.lower() in title.lower()
            tally[name] += hit
            cells.append(f"{'OK ' if hit else '   '}{title[:33]}({readers})")
        print(f"{query:<26}" + "".join(f"{cell:<44}" for cell in cells))

    print()
    for name, _ in conditions:
        print(f"{name:<18} resolved {tally[name]}/{answerable} of the answerable queries")
    print(
        f"\n{len(QUERIES) - answerable} queries are expected to fail under every condition: "
        "L38's finding is that title+author is too thin for a multilingual encoder to "
        "bridge, and no serving rule can fix that."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
