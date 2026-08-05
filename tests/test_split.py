"""Split invariants. This is the leakage test suite.

If any of these break, every metric in ``docs/RESULTS.md`` becomes unciteable, so they
pin the properties the split definition promises rather than just exercising the code.
"""

from __future__ import annotations

import pandas as pd

from recommender.split import make_split


def test_eligibility_rule_is_enforced(toy_ratings: pd.DataFrame) -> None:
    """Only users with >=5 explicit ratings AND >=1 rating >=8 get a holdout."""
    split = make_split(toy_ratings)
    assert sorted(split.test["User-ID"]) == [1, 4, 6]


def test_exactly_one_holdout_per_eligible_user(toy_ratings: pd.DataFrame) -> None:
    split = make_split(toy_ratings)
    assert len(split.test) == split.test["User-ID"].nunique() == 3


def test_holdout_is_always_a_relevant_explicit_rating(toy_ratings: pd.DataFrame) -> None:
    split = make_split(toy_ratings)
    assert (split.test["Book-Rating"] >= split.relevance_threshold).all()


def test_train_and_test_are_disjoint(toy_ratings: pd.DataFrame) -> None:
    """The held-out (user, item) pair must not survive anywhere in train."""
    split = make_split(toy_ratings)
    train_pairs = set(zip(split.train["User-ID"], split.train["ISBN"], strict=True))
    test_pairs = set(zip(split.test["User-ID"], split.test["ISBN"], strict=True))
    assert train_pairs & test_pairs == set()


def test_train_keeps_everything_else(toy_ratings: pd.DataFrame) -> None:
    """Exactly one row per eligible user leaves train -- implicit rows all stay."""
    split = make_split(toy_ratings)
    assert len(split.train) == len(toy_ratings) - len(split.test)
    implicit_before = (toy_ratings["Book-Rating"] == 0).sum()
    implicit_after = (split.train["Book-Rating"] == 0).sum()
    assert implicit_before == implicit_after


def test_same_seed_gives_same_split(toy_ratings: pd.DataFrame) -> None:
    a = make_split(toy_ratings, seed=42)
    b = make_split(toy_ratings, seed=42)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_split_is_independent_of_input_row_order(toy_ratings: pd.DataFrame) -> None:
    """Shuffling the source frame must not change the draw -- otherwise 'seed 42' is
    not a reproducible description of the split."""
    shuffled = toy_ratings.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = make_split(toy_ratings, seed=42).test.sort_values("User-ID").reset_index(drop=True)
    b = make_split(shuffled, seed=42).test.sort_values("User-ID").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_can_give_a_different_holdout(toy_ratings: pd.DataFrame) -> None:
    """User 1 has three relevant books, so some seed must pick a different one."""
    picks = {make_split(toy_ratings, seed=s).holdout_by_user[1] for s in range(12)}
    assert len(picks) > 1
