"""ALS tests. Offline and deterministic: tiny matrix, fixed seed, no downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import BookCrossing, build_interactions
from recommender.models.als import ALSRecommender, confidence_matrix


def test_confidence_is_one_plus_alpha_times_the_grade(toy_ratings: pd.DataFrame) -> None:
    """The pinned signal decision, in one assertion: an ungraded interaction still
    counts (confidence 1), a graded one counts proportionally more."""
    train = build_interactions(toy_ratings)
    confidence = confidence_matrix(train, toy_ratings, alpha=2.0)
    # user 1 rated b3 a 10 -> 1 + 2*10 = 21; user 5 only browsed b1 -> 1 + 2*0 = 1
    assert confidence[train.user_index[1], train.item_index["b3"]] == pytest.approx(21.0)
    assert confidence[train.user_index[5], train.item_index["b1"]] == pytest.approx(1.0)


def test_confidence_keeps_implicit_rows_in_the_matrix(toy_ratings: pd.DataFrame) -> None:
    train = build_interactions(toy_ratings)
    confidence = confidence_matrix(train, toy_ratings, alpha=1.0)
    assert confidence.nnz == train.matrix.nnz


def _fit(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing, **kwargs) -> ALSRecommender:
    train = build_interactions(toy_ratings)
    # The toy books have 1-5 interactions each, so the production support floor of 20
    # would (correctly) reject all of them.
    kwargs.setdefault("similar_min_support", 1)
    model = ALSRecommender(factors=4, iterations=3, seed=42, **kwargs)
    return model.fit(train, toy_catalog, ratings=toy_ratings)


def test_factors_have_the_expected_shape(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    assert model.user_factors.shape == (6, 4)
    assert model.item_factors.shape == (7, 4)


def test_same_seed_gives_the_same_factors(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    a = _fit(toy_ratings, toy_catalog)
    b = _fit(toy_ratings, toy_catalog)
    np.testing.assert_allclose(a.item_factors, b.item_factors)


def test_never_recommends_a_seen_item(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    for user in [1, 2, 3, 5, 6]:
        seen = set(toy_ratings.loc[toy_ratings["User-ID"] == user, "ISBN"])
        picked = {isbn for isbn in model.recommend([user], k=3)[0].tolist() if isbn is not None}
        assert picked & seen == set()


def test_similar_items_excludes_the_query_and_is_ordered(
    toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
) -> None:
    model = _fit(toy_ratings, toy_catalog)
    neighbours = model.similar_items("b1", k=4)
    assert "b1" not in [isbn for isbn, _ in neighbours]
    scores = [score for _, score in neighbours]
    assert scores == sorted(scores, reverse=True)


def test_unknown_isbn_returns_nothing(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    assert _fit(toy_ratings, toy_catalog).similar_items("no-such-isbn") == []


class TestSimilaritySupportFloor:
    """On the real data, 196k items have a single interaction and factors that are noise
    directions; enough of them that one aligns with any query by chance. The floor keeps
    them out of the item-to-item surface."""

    def test_low_support_items_are_excluded_from_neighbourhoods(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        # b7 has a single interaction; b1-b3 have three or more.
        model = _fit(toy_ratings, toy_catalog, similar_min_support=3)
        assert "b7" not in [isbn for isbn, _ in model.similar_items("b1", k=6)]

    def test_floor_of_one_keeps_everything(self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
        model = _fit(toy_ratings, toy_catalog, similar_min_support=1)
        assert "b7" in [isbn for isbn, _ in model.similar_items("b1", k=6)]

    def test_returns_empty_rather_than_noise_when_nothing_clears_the_floor(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        model = _fit(toy_ratings, toy_catalog, similar_min_support=999)
        assert model.similar_items("b1", k=5) == []

    def test_the_floor_does_not_touch_recommendations(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        """It is a product-surface choice, so every ledger metric must be unaffected."""
        loose = _fit(toy_ratings, toy_catalog, similar_min_support=1)
        strict = _fit(toy_ratings, toy_catalog, similar_min_support=999)
        users = [1, 2, 3, 5, 6]
        assert loose.recommend(users, k=3).tolist() == strict.recommend(users, k=3).tolist()


def test_batching_does_not_change_recommendations(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    users = [1, 2, 3, 4, 5, 6]
    big = _fit(toy_ratings, toy_catalog, batch_size=64).recommend(users, k=2)
    small = _fit(toy_ratings, toy_catalog, batch_size=1).recommend(users, k=2)
    assert big.tolist() == small.tolist()
