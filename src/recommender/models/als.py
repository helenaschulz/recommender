"""ALS / weighted matrix factorization — the same story as item-item, one rung up.

This is not an alternative to item-item, it is a continuation of it, and saying so keeps
the model ladder coherent instead of turning it into a bake-off:

- The **item factors are item embeddings**. Similarity is a dot product between them, so
  ALS answers the product's question ("books like this one") exactly as item-item does.
- It gets **personalization for free**. The same fit produces user factors, so the day
  the product knows who is asking, the model already supports it.
- It is the **productionization bridge**. ALS is the one model here with a first-class
  distributed implementation (Spark MLlib), so moving to a cluster is a port rather than a
  rewrite, and it is what an MLflow experiment would track.

**Confidence weighting.** Implicit-feedback ALS treats every stored cell as a confidence
weight on the assertion "this user likes this item". That is where the explicit ratings
earn their place under the pinned signal decision: the matrix carries ``1 + alpha * rating``,
so an ungraded interaction still counts (confidence 1 — it happened, that is evidence)
while a book graded 10 counts ``1 + 10 * alpha`` times as much. Explicit-only models throw
away 62.3% of the rows; this uses all of them and still lets the grades speak.

Note what this does *not* claim: a low rating is not treated as a negative signal. ALS
has no way to express dislike, only strength of positive evidence, and pretending
otherwise would be the kind of quiet modelling error that survives right up until the
A/B test.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recommender.data import BookCrossing, Interactions
from recommender.models.base import Recommender, top_k_from_scores


def confidence_matrix(interactions: Interactions, ratings, alpha: float) -> sp.csr_matrix:
    """Build ``1 + alpha * rating`` over the same index space as *interactions*.

    Args:
        interactions: the binarized train matrix, whose index space is authoritative.
        ratings: the train ratings frame (``User-ID``, ``ISBN``, ``Book-Rating``).
        alpha: how much a full-strength grade outweighs a bare interaction.
    """
    rows = ratings["User-ID"].map(interactions.user_index)
    cols = ratings["ISBN"].map(interactions.item_index)
    keep = rows.notna() & cols.notna()
    grades = ratings.loc[keep, "Book-Rating"].to_numpy(dtype=np.float32)
    confidence = sp.coo_matrix(
        (1.0 + alpha * grades, (rows[keep].to_numpy(dtype=np.int32), cols[keep].to_numpy(dtype=np.int32))),
        shape=interactions.matrix.shape,
        dtype=np.float32,
    ).tocsr()
    confidence.sum_duplicates()
    return confidence


class ALSRecommender(Recommender):
    """Weighted matrix factorization via the ``implicit`` library."""

    name = "als"

    def __init__(
        self,
        *,
        factors: int = 128,
        regularization: float = 0.05,
        iterations: int = 20,
        alpha: float = 1.0,
        similar_min_support: int = 20,
        seed: int = 42,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.similar_min_support = similar_min_support
        self.seed = seed
        self.batch_size = batch_size
        self.params = {
            "factors": factors,
            "regularization": regularization,
            "iterations": iterations,
            "alpha": alpha,
            "similar_min_support": similar_min_support,
            "confidence": "1 + alpha * explicit rating (implicit interactions -> 1)",
            "seed": seed,
        }
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, train: Interactions, catalog: BookCrossing, *, ratings=None) -> ALSRecommender:
        """Fit on train.

        Args:
            ratings: the *train* ratings frame, used only for the confidence weights.
                When omitted the model falls back to the binarized matrix, i.e. every
                interaction gets confidence ``1 + alpha``. It is never read from the
                catalogue, because ``catalog.ratings`` still contains the holdout.
        """
        from implicit.als import AlternatingLeastSquares

        self.train = train
        matrix = (
            confidence_matrix(train, ratings, self.alpha)
            if ratings is not None
            else train.matrix * (1.0 + self.alpha)
        )

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            calculate_training_loss=False,
        )
        model.fit(matrix, show_progress=False)
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        return self

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        train = self._require_fit()
        out = np.full((len(user_ids), k), None, dtype=object)
        rows, targets, blocked = [], [], []
        for row, user_id in enumerate(user_ids):
            user_row = train.user_index.get(int(user_id))
            if user_row is None:
                continue
            lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
            rows.append(user_row)
            targets.append(row)
            blocked.append(train.matrix.indices[lo:hi])

        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start : start + self.batch_size]
            scores = self.user_factors[chunk] @ self.item_factors.T
            picked = top_k_from_scores(scores, k, blocked=blocked[start : start + self.batch_size])
            for local, row in enumerate(targets[start : start + self.batch_size]):
                chosen = picked[local]
                usable = chosen >= 0
                out[row, : usable.sum()] = train.item_ids[chosen[usable]]
        return out

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        """Cosine between item factors, restricted to items with real evidence behind them.

        The support floor is not cosmetic. 196,054 of the 338,496 train items were touched
        exactly once, and their factors are essentially noise directions with tiny norms
        (mean 0.07, against 1.35 for items with 50+ interactions). Individually harmless —
        but with ~196k of them, *some* will align with any query by chance, and in 128
        dimensions the best of 196k coincidences reaches cosine 0.95. Unfiltered, the
        nearest neighbours of *Harry Potter* were five books with one reader each, tied at
        0.941; with the floor they are *The Fellowship of the Ring* and Harry Potter
        volumes 3, 2 and 4. Same factors, same formula — the noise simply outnumbered the
        signal at the argmax.

        This affects only the item-to-item product surface. :meth:`recommend`, and every
        metric in the ledger, is untouched.
        """
        train = self._require_fit()
        item = train.item_index.get(isbn)
        if item is None:
            return []
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1.0
        query = self.item_factors[item] / norms[item]
        scores = (self.item_factors @ query) / norms
        scores[train.item_popularity < self.similar_min_support] = -np.inf
        scores[item] = -np.inf
        take = min(k, scores.size - 1)
        best = np.argpartition(-scores, kth=take - 1)[:take]
        best = best[np.argsort(-scores[best], kind="stable")]
        # Fewer than k items may clear the floor; an empty neighbourhood is the honest
        # answer, never a list padded with excluded items.
        return [(train.item_ids[i], float(scores[i])) for i in best if np.isfinite(scores[i])]
