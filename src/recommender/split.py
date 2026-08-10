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

**The anchor draw, pinned once.** :func:`pick_anchors` adds a second draw to this
module, and it deliberately reuses the first one's mechanism. The item-to-item metric needs
one *query* book per user — the book a visitor would paste in — and the honest choice is the
book we would have held out instead: drawn uniformly, seeded, from the same population the
holdout came from (that user's train ratings >= ``relevance_threshold``). Same population,
same mechanism, same seed. It lives here rather than in the metric because a draw that
decides a published number belongs next to the other draw that does, where the two can be
read against each other, and because the anchor rule is a **free parameter** — two
defensible rules give two different numbers, so the rule travels with the number.

**Leakage discipline.** This module is the only place a holdout is chosen. Everything
downstream — item popularity, similarities, factors, IDF statistics — is computed from
:attr:`Split.train` and never from :attr:`Split.test`. That is the property the tests in
``tests/test_split.py`` pin down, because item-item PoCs classically fail exactly here,
by computing similarities on the full matrix before splitting. The anchor draw obeys the
same rule from the other side: it reads ``split.train`` only, so an anchor can never *be*
the held-out item and filtering a reader's train items out of a neighbour list can never
remove the target.
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


@dataclass(frozen=True)
class Anchors:
    """One query item per eligible user, aligned row-for-row with :attr:`Split.test`.

    The row alignment is the whole contract. It is what lets the item-to-item column be
    compared to the profile column user by user under
    :func:`recommender.eval.mcnemar` — the same readers, the same held-out books, only the
    question changes. A permuted pair does not raise anywhere; it quietly tests noise.
    """

    frame: pd.DataFrame  # User-ID, ISBN, Book-Rating, fallback — in split.test row order
    seed: int
    relevance_threshold: int

    @property
    def item_ids(self) -> np.ndarray:
        return self.frame["ISBN"].to_numpy()

    @property
    def user_ids(self) -> np.ndarray:
        return self.frame["User-ID"].to_numpy()

    @property
    def n_fallback(self) -> int:
        """Users whose held-out book took their only rating >= the threshold."""
        return int(self.frame["fallback"].sum())

    def describe(self) -> str:
        return (
            f"one anchor per user, seed={self.seed}: drawn uniformly from that reader's train "
            f"ratings >={self.relevance_threshold}, falling back to any graded train rating when "
            f"the holdout took the only one. {len(self.frame):,} anchors, "
            f"{self.n_fallback:,} ({self.n_fallback / max(len(self.frame), 1):.1%}) from the fallback pool."
        )


def pick_anchors(
    split: Split,
    *,
    seed: int = SEED,
    relevance_threshold: int | None = None,
) -> Anchors:
    """Draw one anchor item per eligible user from that user's **train** rows.

    The preferred pool is the reader's train ratings >= *relevance_threshold*, i.e. the
    same population :func:`make_split` drew the holdout from. Readers whose only such
    rating *was* the holdout fall back to any graded train row; they are flagged in the
    ``fallback`` column and counted in :meth:`Anchors.describe`, because a fallback anchor
    is a book the reader graded but did not love, and a table that hid how many there were
    would be hiding a choice.

    Deterministic in *seed* and independent of input row order: the candidate frame is
    sorted before the draw, exactly as in :func:`make_split`. The seed is a free parameter
    of the measurement, so it is reported with it and a second seed is run as a labelled
    sensitivity check rather than as a result.
    """
    threshold = split.relevance_threshold if relevance_threshold is None else relevance_threshold
    eligible = set(split.test["User-ID"])
    explicit = split.train.loc[(split.train["Book-Rating"] > 0) & split.train["User-ID"].isin(eligible)]
    preferred = explicit.loc[explicit["Book-Rating"] >= threshold]
    covered = set(preferred["User-ID"])
    pool = pd.concat([preferred, explicit.loc[~explicit["User-ID"].isin(covered)]])
    # Sort first so the draw depends only on the seed, never on input row order.
    pool = pool.sort_values(["User-ID", "ISBN"], kind="mergesort")

    rng = np.random.default_rng(seed)
    if pool.empty:
        picked = np.array([], dtype=split.train.index.dtype)
    else:
        picked = (
            pool.groupby("User-ID", sort=True)
            .apply(lambda g: g.index[rng.integers(len(g))], include_groups=False)
            .to_numpy()
        )

    frame = (
        split.train.loc[picked, ["User-ID", "ISBN", "Book-Rating"]]
        .assign(fallback=lambda f: ~f["User-ID"].isin(covered))
        .set_index("User-ID")
        .reindex(split.test["User-ID"])  # the load-bearing line: row order becomes split.test's
        .reset_index()
    )
    if frame["ISBN"].isna().any():
        # Unreachable from make_split: eligibility needs >=5 explicit ratings and exactly
        # one is held out, so >=4 graded train rows always remain. Reachable by handing in
        # a Split built some other way, and a NaN anchor would score as a silent miss for
        # every model rather than as an error.
        missing = int(frame["ISBN"].isna().sum())
        raise ValueError(f"{missing} eligible users have no graded train row to draw an anchor from")
    return Anchors(frame=frame, seed=seed, relevance_threshold=threshold)
