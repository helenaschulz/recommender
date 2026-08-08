"""Serving layer: what happens to a model's output between the model and the screen.

Right now it holds one thing, and one thing is enough to justify the layer existing.

**Work-level deduplication.** The models score ISBNs, because that is what the
interaction data is keyed on. A reader does not want ISBNs. Ledger L31 measured the
consequence: 31.6% of the TF-IDF model's recommendation slots are another *edition* of a
book already in the user's profile, and 73.4% of users see at least one — the top seven
neighbours of *The Da Vinci Code* are seven ISBNs of *The Da Vinci Code*. L39 found the
same failure on the embedding model's similarity surface.

This is deliberately **not** fixed inside the models, and not by re-keying the whole
pipeline to works. Two reasons:

- The models are unchanged and every number in the comparison table still means what it
  meant. Deduplication is a presentation decision, so it belongs where presentation
  decisions belong, and it can be switched off to measure exactly what it is worth.
- The work-level *experiment* (ledger L44) is a different question — does clustering
  before training lift accuracy — and mixing the two would make neither answerable.

The wrapper asks the inner model for ``oversample * k`` candidates, then keeps the first
one per work, skipping works the user already has. Over-fetching is what makes a full
top-k still possible after a heavily duplicated list is collapsed; where even that is not
enough, the remaining slots stay ``None`` rather than being padded with a duplicate.
"""

from __future__ import annotations

import numpy as np

from recommender.data import BookCrossing, Interactions, Works
from recommender.models.base import Recommender

#: Enough to survive the worst measured case: the catalogue holds 120 Harry Potter rows,
#: and 8 of TF-IDF's top 10 for that anchor were editions of the anchor itself.
DEFAULT_OVERSAMPLE = 10


class WorkDeduped(Recommender):
    """Collapse an ISBN-level model's output to one ISBN per work, at serving time."""

    def __init__(self, inner: Recommender, works: Works, *, oversample: int = DEFAULT_OVERSAMPLE) -> None:
        super().__init__()
        self.inner = inner
        self.works = works
        self.oversample = oversample
        self.name = f"{inner.name} + work-dedup"
        self.params = dict(inner.params, serving="work-level dedup", oversample=oversample)
        # Wrapping an already-fitted model is the normal case: dedup is a serving
        # decision taken after training, so it must not require a re-fit.
        self.train = inner.train

    def fit(self, train: Interactions, catalog: BookCrossing) -> WorkDeduped:
        self.inner.fit(train, catalog)
        self.train = self.inner.train
        return self

    def describe_params(self) -> str:
        return f"{self.inner.describe_params()}; serving: work-dedup (oversample={self.oversample})"

    def train_works(self, user_row: int) -> set[str]:
        """The works a user already interacted with in train — what dedup blocks."""
        train = self._require_fit()
        lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
        return set(self.works.of(train.item_ids[train.matrix.indices[lo:hi]]).tolist())

    def collapse(self, candidates: np.ndarray, user_ids: np.ndarray, k: int) -> np.ndarray:
        """Reduce an already-scored ``(users, n)`` candidate block to ``(users, k)``.

        Split out of :meth:`recommend` so ``scripts/analyze_dedup.py`` can apply the filter
        to a candidate block it scored itself. The point is that there is exactly one
        implementation of the rule: the "after dedup" number in the ledger is produced by
        the same code the app would serve from, not by a measurement-only copy of it.
        """
        train = self._require_fit()
        out = np.full((len(user_ids), k), None, dtype=object)
        for row, user in enumerate(user_ids):
            usable = [isbn for isbn in candidates[row] if isbn is not None]
            if not usable:
                continue
            user_row = train.user_index.get(int(user))
            blocked = self.train_works(user_row) if user_row is not None else set()
            kept: list[str] = []
            for isbn, work in zip(usable, self.works.of(usable), strict=True):
                if work in blocked:
                    continue
                blocked.add(work)
                kept.append(isbn)
                if len(kept) == k:
                    break
            out[row, : len(kept)] = kept
        return out

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        return self.collapse(self.inner.recommend(user_ids, k=k * self.oversample), user_ids, k)

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        neighbours = self.inner.similar_items(isbn, k=k * self.oversample)
        if not neighbours:
            return []
        seen = {self.works.of([isbn])[0]}
        kept: list[tuple[str, float]] = []
        others = [other for other, _ in neighbours]
        for (other, score), work in zip(neighbours, self.works.of(others), strict=True):
            if work in seen:
                continue
            seen.add(work)
            kept.append((other, score))
            if len(kept) == k:
                break
        return kept


def duplicate_slot_rate(
    recommended: np.ndarray,
    user_ids: np.ndarray,
    train: Interactions,
    works: Works,
) -> dict[str, float]:
    """Ledger L31's metric: how much of a top-k list is a book the user already has.

    Returns the share of *slots* that are another edition of something in the user's train
    profile, and the share of *users* who get at least one. Both are needed: a 30% slot
    rate concentrated on a few users is a different product problem from one spread over
    everybody.
    """
    filled = 0
    duplicate = 0
    affected = 0
    for row, user in enumerate(user_ids):
        user_row = train.user_index.get(int(user))
        if user_row is None:
            continue
        lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
        owned = set(works.of(train.item_ids[train.matrix.indices[lo:hi]]).tolist())
        slots = [isbn for isbn in recommended[row] if isbn is not None]
        if not slots:
            continue
        hits = sum(work in owned for work in works.of(slots))
        filled += len(slots)
        duplicate += hits
        affected += hits > 0
    return {
        "duplicate_slot_share": duplicate / filled if filled else float("nan"),
        "affected_user_share": affected / len(user_ids) if len(user_ids) else float("nan"),
        "filled_slots": float(filled),
    }
