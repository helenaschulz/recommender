"""The one interface every candidate model implements.

Two methods, because there are two jobs here that are easy to conflate:

- :meth:`Recommender.recommend` — *offline evaluation*. Given users, return the top-k
  items for each, excluding what they already interacted with in train. This is what
  HitRate@10 measures.
- :meth:`Recommender.similar_items` — *the product*. Given one book, return books like
  it. This is what the app does, and what the face-validity gallery shows.

The offline harness scores user histories; the product is item-to-item. That gap is
deliberate and is argued in ``docs/model_selection.md`` rather than papered over: a model
that ranks a user's next book well is evidence about its item neighbourhoods, not proof.

Implementations return **ISBNs**, never internal column indices. Each model owns its own
candidate universe (collaborative models can only reach items that appear in train;
content models reach the whole catalogue), so translating indices is the model's job and
the evaluation layer stays free of index bookkeeping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from recommender.data import BookCrossing, Interactions


class Recommender(ABC):
    """Common interface: fit on a train matrix, then recommend for users or items."""

    #: Short label used in ledger rows, galleries and the comparison table.
    name: str = "recommender"

    def __init__(self) -> None:
        self.train: Interactions | None = None
        self.params: dict[str, Any] = {}

    @abstractmethod
    def fit(self, train: Interactions, catalog: BookCrossing) -> Recommender:
        """Fit on train interactions only. Never sees the holdout."""

    @abstractmethod
    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        """Top-k ISBNs per user, excluding each user's own train items.

        Returns an object array of shape ``(len(user_ids), k)``. Where a model cannot
        fill k slots for a user (too few candidates), the remaining entries are ``None``.
        """

    @abstractmethod
    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        """The k most similar items to *isbn*, as ``(isbn, score)``, best first.

        Returns an empty list when the model has no representation for *isbn* — an
        honest empty answer beats a fabricated neighbourhood.
        """

    def recommend_scored(self, user_ids: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """The same top-k as :meth:`recommend`, with the scores that produced it.

        Returns ``(ids, scores)``, both of shape ``(len(user_ids), k)``: ids as an object
        array padded with ``None``, scores as float64 padded with ``-inf``. Optional —
        models that cannot produce a comparable score raise, and the default here does.

        **This exists for the hybrid (milestone M19) and for nothing else.** Combining two
        rankers needs more than their top-10 lists: score fusion needs the scores, and every
        rule needs a deeper candidate list than the ten slots that end up on screen. Where a
        model implements it, :meth:`recommend` is a thin slice of it rather than a second
        ranking path — one code path, so a hybrid can never be fed a ranking that differs
        from the one the comparison table measured.

        **The scores are not comparable across models and this method does not pretend
        otherwise.** Item-item returns a *sum* of shrunk cosines over the user's profile,
        TF-IDF a *mean* cosine; the first grows with profile length and the second does not.
        Normalizing them against each other is the caller's problem, and the caller has to
        say what it did (see :mod:`recommender.models.hybrid`).
        """
        raise NotImplementedError(f"{type(self).__name__} does not expose scores; see Recommender.recommend_scored")

    def _require_fit(self) -> Interactions:
        if self.train is None:
            raise RuntimeError(f"{type(self).__name__}.fit() must be called before use")
        return self.train

    def describe_params(self) -> str:
        """One-line parameter record for the ledger. Every unrecorded parameter is a
        number nobody can reproduce later, so models fill this in."""
        return ", ".join(f"{key}={value}" for key, value in self.params.items()) or "none"


def top_k_from_scores(
    scores: np.ndarray,
    k: int,
    *,
    blocked: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Row-wise top-k column indices of a dense score block, best first.

    ``blocked[i]`` lists column indices to exclude for row ``i`` (a user's train items).
    Excluded and non-finite entries are pushed to -inf before selection, so they can
    never be recommended. Rows with fewer than k usable candidates get ``-1`` padding.
    """
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[~np.isfinite(scores)] = -np.inf
    if blocked is not None:
        for row, cols in enumerate(blocked):
            if len(cols):
                scores[row, cols] = -np.inf

    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-ordered, axis=1, kind="stable")
    picked = np.take_along_axis(part, order, axis=1)
    # A -inf score means "no candidate left", not "rank 10".
    usable = np.take_along_axis(scores, picked, axis=1) > -np.inf
    return np.where(usable, picked, -1)
