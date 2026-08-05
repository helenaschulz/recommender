"""Content-model tests. Offline: the vectorizer is fitted on the seven toy titles."""

from __future__ import annotations

import pandas as pd

from recommender.data import BookCrossing, build_interactions
from recommender.models.content_tfidf import TfidfRecommender, content_text


def _fit(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> TfidfRecommender:
    return TfidfRecommender(min_df=1).fit(build_interactions(toy_ratings), toy_catalog)


def test_content_text_joins_title_and_author_lowercased(toy_books: pd.DataFrame) -> None:
    assert content_text(toy_books)[0] == "the hobbit j.r.r. tolkien"


def test_content_text_survives_missing_metadata() -> None:
    books = pd.DataFrame({"Book-Title": ["Solo", None], "Book-Author": [None, "Nobody"]})
    assert content_text(books).tolist() == ["solo", "nobody"]


def test_same_author_books_are_nearest_neighbours(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    """The Hobbit's neighbours should be the other two Tolkien books, on text alone."""
    model = _fit(toy_ratings, toy_catalog)
    neighbours = [isbn for isbn, _ in model.similar_items("b1", k=2)]
    assert set(neighbours) == {"b2", "b3"}


def test_an_item_is_never_its_own_neighbour(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    assert "b1" not in [isbn for isbn, _ in model.similar_items("b1", k=6)]


def test_reaches_books_with_zero_interactions(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    """The whole point of the coverage layer: a book nobody touched is still scoreable.

    ``b7`` is interacted with only by user 4, so for every other user it is a candidate
    that collaborative filtering could rank only by accident.
    """
    model = _fit(toy_ratings, toy_catalog)
    assert model.similar_items("b7", k=3) != []
    reachable = {isbn for row in model.recommend([1, 2, 3, 5, 6], k=6) for isbn in row.tolist() if isbn}
    assert "b7" in reachable


def test_never_recommends_a_seen_item(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    for user in [1, 2, 3, 4, 5, 6]:
        seen = set(toy_ratings.loc[toy_ratings["User-ID"] == user, "ISBN"])
        picked = {isbn for isbn in model.recommend([user], k=6)[0].tolist() if isbn is not None}
        assert picked & seen == set()


def test_unknown_isbn_returns_nothing_rather_than_guessing(
    toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
) -> None:
    model = _fit(toy_ratings, toy_catalog)
    assert model.similar_items("no-such-isbn") == []


def test_batching_does_not_change_recommendations(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    train = build_interactions(toy_ratings)
    big = TfidfRecommender(min_df=1, batch_size=64).fit(train, toy_catalog)
    small = TfidfRecommender(min_df=1, batch_size=1).fit(train, toy_catalog)
    users = [1, 2, 3, 4, 5, 6]
    assert big.recommend(users, k=3).tolist() == small.recommend(users, k=3).tolist()
