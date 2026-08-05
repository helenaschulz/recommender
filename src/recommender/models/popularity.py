"""Popularity baseline: recommend the most-interacted-with books, minus what you've seen.

This is the reference row. Not because it is a good recommender — it is not personalized
at all, it hands almost the same ten books to everyone — but because it calibrates every
other number in the ledger. On a catalogue where the top 1% of books absorb 25.1% of all
interactions (ledger L9), "recommend the bestsellers" is a genuinely hard accuracy
benchmark, and a model that cannot beat it has learned nothing that popularity did not
already know.

It is also the clearest illustration of why HitRate alone is not a verdict: this model
should score a respectable HitRate and a *terrible* Coverage@10, because it can only
ever surface a few dozen distinct titles across the entire user base.

The only per-user element is exclusion: a user is never recommended a book they already
interacted with in train.
"""

from __future__ import annotations

import numpy as np

from recommender.data import BookCrossing, Interactions
from recommender.models.base import Recommender


class PopularityRecommender(Recommender):
    """Rank items by train interaction count, identical for every user."""

    name = "popularity"

    def __init__(self, candidate_pool: int = 2000) -> None:
        """Args:
        candidate_pool: how many top titles to keep as candidates. Only needs to
            exceed k + the largest train profile among evaluated users so that
            exclusion can never exhaust the list; 2,000 is far above both.
        """
        super().__init__()
        self.candidate_pool = candidate_pool
        self.params = {"candidate_pool": candidate_pool}
        self._ranking: np.ndarray | None = None

    def fit(self, train: Interactions, catalog: BookCrossing) -> PopularityRecommender:
        self.train = train
        popularity = train.item_popularity
        top = np.argsort(-popularity, kind="stable")[: self.candidate_pool]
        self._ranking = top[popularity[top] > 0]
        return self

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        train = self._require_fit()
        ranking = self._ranking
        out = np.full((len(user_ids), k), None, dtype=object)
        candidates = ranking.tolist()
        for row, user_id in enumerate(user_ids):
            user_row = train.user_index.get(int(user_id))
            seen: set[int] = set()
            if user_row is not None:
                start, end = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
                seen = set(train.matrix.indices[start:end].tolist())
            picked = [item for item in candidates if item not in seen][:k]
            out[row, : len(picked)] = train.item_ids[picked]
        return out

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        """Popularity has no notion of similarity.

        The gallery still shows this column, labelled as what it is: the global top-10,
        the same answer for every anchor. It is the visual control that makes the other
        models' neighbourhoods mean something.
        """
        train = self._require_fit()
        picked = [item for item in self._ranking.tolist() if train.item_ids[item] != isbn][:k]
        return [(train.item_ids[item], float(train.item_popularity[item])) for item in picked]
