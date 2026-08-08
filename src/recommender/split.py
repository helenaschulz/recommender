"""Evaluation split. One definition, pinned once, cited by every model line in the ledger.

**The split, pinned once.** Per-user leave-one-out:

- *Eligible user*: has at least ``min_explicit`` explicit ratings **and** at least one
  explicit rating >= ``relevance_threshold``. The first condition means the user has a
  profile left after the holdout; the second means there is something worth predicting.
- *Holdout*: exactly one item per eligible user, drawn seeded-at-random from that user's
  explicit ratings >= ``relevance_threshold``.
- *Train*: literally everything else, including every implicit interaction of every user
  and all interactions of users who are not eligible. Sparse data is not a reason to
  throw signal away.

Why leave-one-out and not a temporal split: ``Ratings.csv`` has no time column at all
(ledger L18), so "train on the past, test on the future" is not expressible on this
dataset. With real production logs the split would be temporal, and a random split there
would leak the future into training.

**Leakage discipline.** This module is the only place a holdout is chosen. Everything
downstream — item popularity, similarities, factors, IDF statistics — is computed from
:attr:`Split.train` and never from :attr:`Split.test`. That is the property the tests in
``tests/test_split.py`` pin down, because item-item PoCs classically fail exactly here,
by computing similarities on the full matrix before splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 42
MIN_EXPLICIT = 5
RELEVANCE_THRESHOLD = 8


@dataclass(frozen=True)
class Split:
    """A leave-one-out split plus the parameters that produced it."""

    train: pd.DataFrame
    test: pd.DataFrame  # exactly one row per eligible user: User-ID, ISBN, Book-Rating
    seed: int
    min_explicit: int
    relevance_threshold: int

    @property
    def n_eligible(self) -> int:
        return len(self.test)

    @property
    def holdout_by_user(self) -> dict[int, str]:
        return dict(zip(self.test["User-ID"], self.test["ISBN"], strict=True))

    def describe(self) -> str:
        return (
            f"leave-one-out, seed={self.seed}: eligible = users with >={self.min_explicit} explicit "
            f"ratings and >=1 rating >={self.relevance_threshold}; one held-out item per eligible "
            f"user drawn from their ratings >={self.relevance_threshold}. "
            f"{self.n_eligible:,} eligible users, {len(self.train):,} train interactions."
        )


def make_split(
    ratings: pd.DataFrame,
    *,
    seed: int = SEED,
    min_explicit: int = MIN_EXPLICIT,
    relevance_threshold: int = RELEVANCE_THRESHOLD,
) -> Split:
    """Build the pinned leave-one-out split.

    Deterministic in *seed*: the same seed on the same frame always yields the same
    holdout, which is what makes model rows comparable across milestones.
    """
    explicit = ratings.loc[ratings["Book-Rating"] > 0]
    n_explicit = explicit.groupby("User-ID").size()
    relevant = explicit.loc[explicit["Book-Rating"] >= relevance_threshold]

    eligible = np.intersect1d(
        n_explicit.index[n_explicit >= min_explicit].to_numpy(),
        relevant["User-ID"].unique(),
    )

    candidates = relevant.loc[relevant["User-ID"].isin(set(eligible))]
    # Sort first so the draw depends only on the seed, never on input row order.
    candidates = candidates.sort_values(["User-ID", "ISBN"], kind="mergesort")
    rng = np.random.default_rng(seed)
    if candidates.empty:
        # No eligible user. Legitimate on a thin frame — a validation split carved out of
        # an already-thin train set can run out of users — and an empty holdout is the
        # honest answer. Without this, groupby.apply returns an empty *frame* and the
        # .loc below fails with "Cannot index with multidimensional key", which says
        # nothing about the actual cause.
        picked = np.array([], dtype=ratings.index.dtype)
    else:
        picked = (
            candidates.groupby("User-ID", sort=True)
            .apply(lambda g: g.index[rng.integers(len(g))], include_groups=False)
            .to_numpy()
        )

    test = ratings.loc[picked, ["User-ID", "ISBN", "Book-Rating"]].reset_index(drop=True)
    train = ratings.drop(index=picked)
    return Split(
        train=train,
        test=test,
        seed=seed,
        min_explicit=min_explicit,
        relevance_threshold=relevance_threshold,
    )
