"""Face-validity gallery: the same three books, every model, side by side.

Why this exists. The offline harness measures how well a model
ranks a user's next book. The product does something different: it takes one book and
returns books like it, with no user identity at query time. A HitRate cannot see whether
the neighbourhoods a model builds are *sensible*, and a plausible-looking neighbourhood
cannot see whether they are *predictive*. Reporting both, and saying where each one is
blind, is cheaper than defending one number that quietly stands in for the other.

The anchors are fixed so every model is judged on identical input:

- ``0385504209`` **The Da Vinci Code** — the 2003 blockbuster, 853 train interactions.
  Dense, popular, mainstream: the easy case. If a model cannot do this one, stop.
- ``043936213X`` **Harry Potter and the Sorcerer's Stone (Book 1)** — 101 train
  interactions. The diagnostic anchor: a working item-item model should surface the
  other Harry Potter volumes, and the catalogue holds 120 Harry Potter rows across
  editions and languages, so it also exposes the edition-duplication problem (ledger
  L15) in a way a metric never will.
- ``0316666343`` **The Lovely Bones: A Novel** — 1,248 train interactions. Literary
  fiction rather than genre, to check the models are not just reproducing one cluster.

Chosen for recognisability to a non-specialist reader and for high train support, so a
weak result is the model's fault and not a data-sparsity accident. Deliberately *not*
chosen: ``0971880107`` *Wild Animus*, the most-interacted title in the whole dataset —
its author gave it away by the crate on BookCrossing, so its co-occurrence structure
says more about a marketing campaign than about reading taste.
"""

from __future__ import annotations

import pandas as pd

from recommender.data import BookCrossing
from recommender.models.base import Recommender

ANCHORS: dict[str, str] = {
    "0385504209": "The Da Vinci Code",
    "043936213X": "Harry Potter and the Sorcerer's Stone",
    "0316666343": "The Lovely Bones",
}

#: The demo app's buttons, in presentation order (M14.8, Helena's decision). Label -> work
#: id. It lives here rather than in ``app/main.py`` because two other things need the same
#: list — ``scripts/capture_app_screenshots.py`` drives these buttons by label, and it
#: silently timed out the first time the labels changed in only one of the two places.
#:
#: The order is the argument: the trivial case first (three of *The Da Vinci Code*'s top
#: four are Dan Brown), then the series case, then the payoff (*To Kill a Mockingbird*
#: reaches Steinbeck, Golding, Kesey with **zero** same-author hits), then the
#: counterexample — *Fight Club* has 102 interactions and an excellent list, which is what
#: stops L63's support story from collapsing into "less data = worse".
DEMO_BUTTONS: dict[str, str] = {
    "The Da Vinci Code": "the da vinci code|brown",
    "Harry Potter": "harry potter and the sorcerer's stone|rowling",
    "To Kill a Mockingbird": "to kill a mockingbird|lee",
    "Fight Club": "fight club|palahniuk",
}

#: The demo's reading set (M14), keyed by **work id** because the app's item is a work.
#:
#: The three above are chosen to compare *models* on identical input. These eleven are
#: chosen to read one model's *output* — which is what found M14 at all, since no cell in
#: the comparison table shows a missing author tag, a duplicate at rank 2, or a noise slot
#: at rank 9. Ten are the anchors Helena's manual pass named; *Dune* is Cody's addition,
#: because the other ten are literary fiction, thriller and fantasy and a set with no
#: science fiction in it cannot notice a genre-specific failure.
#:
#: The set deliberately spans the support bands `scripts/analyze_anchor_support.py`
#: measures, from *Guns, Germs, and Steel* at 67 interactions to *The Lovely Bones* at
#: 1,295. A uniformly well-supported anchor set would hide the calibration problem.
DEMO_ANCHORS: dict[str, str] = {
    "the da vinci code|brown": "The Da Vinci Code",
    "harry potter and the sorcerer's stone|rowling": "Harry Potter and the Sorcerer's Stone",
    "the lovely bones: a novel|sebold": "The Lovely Bones",
    "to kill a mockingbird|lee": "To Kill a Mockingbird",
    "girl with a pearl earring|chevalier": "Girl with a Pearl Earring",
    "interview with the vampire|rice": "Interview with the Vampire",
    "fight club|palahniuk": "Fight Club",
    "bridget jones's diary|fielding": "Bridget Jones's Diary",
    # No space before the colon: these ids follow the **serving** key, which carries the
    # M14.4 punctuation fix (ledger L64). Under the published M11 key this one reads
    # "the hobbit : the enchanting prelude ...".
    "the hobbit: the enchanting prelude to the lord of the rings|tolkien": "The Hobbit",
    "guns, germs, and steel: the fates of human societies|diamond": "Guns, Germs, and Steel",
    "dune|herbert": "Dune",
}


def similar_books(model: Recommender, isbn: str, catalog: BookCrossing, k: int = 10) -> list[str]:
    """Human-readable top-k neighbours of *isbn* under *model*."""
    neighbours = model.similar_items(isbn, k=k)
    if not neighbours:
        return [f"— no representation for {isbn} —"]
    return [f"{catalog.describe(other)}  ({score:.3f})" for other, score in neighbours]


def build_gallery(
    models: list[Recommender],
    catalog: BookCrossing,
    *,
    anchors: dict[str, str] | None = None,
    k: int = 10,
) -> dict[str, pd.DataFrame]:
    """One table per anchor: columns are models, rows are ranks 1..k."""
    anchors = anchors or ANCHORS
    tables: dict[str, pd.DataFrame] = {}
    for isbn, label in anchors.items():
        columns = {}
        for model in models:
            rows = similar_books(model, isbn, catalog, k=k)
            columns[model.name] = rows + [""] * (k - len(rows))
        tables[label] = pd.DataFrame(columns, index=pd.RangeIndex(1, k + 1, name="rank"))
    return tables


def render_markdown(tables: dict[str, pd.DataFrame]) -> str:
    """Markdown rendering for docs and PR descriptions."""
    parts = []
    for label, table in tables.items():
        parts.append(f"### Similar to *{label}*\n")
        parts.append(table.to_markdown())
        parts.append("")
    return "\n".join(parts)
