"""Popularity baseline behaviour, on the toy fixture where the ranking is countable by eye.

Train interaction counts in ``toy_ratings``:
``b1``:5, ``b2``:5, ``b3``:5, ``b4``:4, ``b5``:3, ``b6``:2, ``b7``:1.
"""

from __future__ import annotations

import pandas as pd

from recommender.data import BookCrossing, build_interactions
from recommender.models.popularity import PopularityRecommender


def _fit(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> PopularityRecommender:
    return PopularityRecommender().fit(build_interactions(toy_ratings), toy_catalog)


def test_ranks_by_interaction_count(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    # User 5 has only touched b1, b2, b3, so the next most popular are b4, b5, b6.
    assert model.recommend([5], k=3)[0].tolist() == ["b4", "b5", "b6"]


def test_never_recommends_a_users_own_train_items(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    seen = set(toy_ratings.loc[toy_ratings["User-ID"] == 1, "ISBN"])
    assert set(model.recommend([1], k=7)[0].tolist()) & seen == set()


def test_pads_with_none_when_candidates_run_out(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    """User 4 has touched six of seven books, so only b6 is left; the rest stay empty."""
    got = model_row = _fit(toy_ratings, toy_catalog).recommend([4], k=3)[0].tolist()
    assert model_row[0] == "b6"
    assert got[1:] == [None, None]


def test_ranking_is_identical_for_every_user(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    """The point of the baseline: it is not personalized. Only exclusion differs."""
    model = _fit(toy_ratings, toy_catalog)
    fresh_ratings = pd.DataFrame({"User-ID": [99, 98], "ISBN": ["b7", "b7"], "Book-Rating": [5, 5]})
    model_all = PopularityRecommender().fit(
        build_interactions(pd.concat([toy_ratings, fresh_ratings])), toy_catalog
    )
    assert model_all.recommend([99], k=3)[0].tolist() == model_all.recommend([98], k=3)[0].tolist()
    assert model.recommend([5], k=2)[0].tolist() == ["b4", "b5"]


def test_similar_items_returns_the_global_top_and_never_the_query(
    toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
) -> None:
    model = _fit(toy_ratings, toy_catalog)
    neighbours = [isbn for isbn, _ in model.similar_items("b1", k=3)]
    assert "b1" not in neighbours
    assert neighbours == ["b2", "b3", "b4"]
