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
