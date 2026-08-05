"""Item-item similarity tests, including the shrinkage property the model exists for."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recommender.data import BookCrossing, build_interactions
from recommender.models.item_item import ItemItemRecommender, shrunk_cosine_neighbours


def test_cosine_without_shrinkage_matches_the_hand_computed_value() -> None:
    """Three users; items 0 and 1 co-occur twice, each has support 2 and 3.

    cos(0, 1) = 2 / sqrt(2 * 3) = 0.8165
    """
    matrix = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32))
    sim = shrunk_cosine_neighbours(matrix, shrinkage=0.0, top_k=10)
    assert sim[0, 1] == pytest.approx(2 / np.sqrt(2 * 3), rel=1e-5)


def test_shrinkage_divides_by_support_plus_lambda() -> None:
    matrix = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32))
    sim = shrunk_cosine_neighbours(matrix, shrinkage=10.0, top_k=10)
    assert sim[0, 1] == pytest.approx(2 / (np.sqrt(2 * 3) + 10.0), rel=1e-5)


def test_shrinkage_demotes_a_one_off_pair_below_a_well_supported_one() -> None:
    """The whole reason shrinkage is in this model.

    Items 0 and 1 share four readers out of four each -> plain cosine 1.0.
    Items 2 and 3 share their single reader -> plain cosine also 1.0.
    Unshrunk, a coincidence ties with strong evidence; shrunk, it must not.
    """
    dense = np.zeros((5, 4), dtype=np.float32)
    dense[0:4, 0] = 1
    dense[0:4, 1] = 1
    dense[4, 2] = 1
    dense[4, 3] = 1
    matrix = sp.csr_matrix(dense)

    plain = shrunk_cosine_neighbours(matrix, shrinkage=0.0, top_k=10)
    assert plain[0, 1] == pytest.approx(plain[2, 3])

    shrunk = shrunk_cosine_neighbours(matrix, shrinkage=10.0, top_k=10)
    assert shrunk[0, 1] > shrunk[2, 3]


def test_diagonal_is_zero_so_an_item_is_never_its_own_neighbour() -> None:
    matrix = sp.csr_matrix(np.array([[1, 1], [1, 1]], dtype=np.float32))
    sim = shrunk_cosine_neighbours(matrix, shrinkage=0.0, top_k=5)
    assert sim[0, 0] == 0.0
    assert sim[1, 1] == 0.0


def test_top_k_truncation_keeps_only_the_strongest_neighbours() -> None:
    rng = np.random.default_rng(0)
    matrix = sp.csr_matrix((rng.random((40, 12)) < 0.5).astype(np.float32))
    sim = shrunk_cosine_neighbours(matrix, shrinkage=1.0, top_k=3)
    assert sim.getrow(0).nnz <= 3


def test_min_support_drops_items_below_the_threshold() -> None:
    dense = np.zeros((4, 3), dtype=np.float32)
    dense[0:3, 0] = 1
    dense[0:3, 1] = 1
    dense[0, 2] = 1  # support 1
    sim = shrunk_cosine_neighbours(sp.csr_matrix(dense), shrinkage=0.0, top_k=5, min_support=2)
    assert sim.getrow(2).nnz == 0
    assert sim[:, 2].nnz == 0


def test_blocking_does_not_change_the_result() -> None:
    """The blocked loop is an implementation detail; the matrix must not depend on it."""
    rng = np.random.default_rng(1)
    matrix = sp.csr_matrix((rng.random((30, 20)) < 0.4).astype(np.float32))
    whole = shrunk_cosine_neighbours(matrix, shrinkage=5.0, top_k=4, block=1000)
    blocked = shrunk_cosine_neighbours(matrix, shrinkage=5.0, top_k=4, block=3)
    assert (whole != blocked).nnz == 0


class TestRecommend:
    def test_recommends_neighbours_of_what_the_user_read(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        train = build_interactions(toy_ratings)
        model = ItemItemRecommender(shrinkage=0.0, top_k_neighbours=10).fit(train, toy_catalog)
        picked = model.recommend([5], k=3)[0].tolist()
        assert picked[0] is not None
        seen = set(toy_ratings.loc[toy_ratings["User-ID"] == 5, "ISBN"])
        assert set(picked) & seen == set()

    def test_never_recommends_a_seen_item(self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
        train = build_interactions(toy_ratings)
        model = ItemItemRecommender(shrinkage=0.0, top_k_neighbours=10).fit(train, toy_catalog)
        for user in [1, 2, 3, 4, 5, 6]:
            seen = set(toy_ratings.loc[toy_ratings["User-ID"] == user, "ISBN"])
            picked = {isbn for isbn in model.recommend([user], k=7)[0].tolist() if isbn is not None}
            assert picked & seen == set()

    def test_unknown_item_yields_an_empty_neighbourhood_not_a_guess(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        model = ItemItemRecommender().fit(build_interactions(toy_ratings), toy_catalog)
        assert model.similar_items("not-an-isbn") == []

    def test_similar_items_are_ordered_by_score(self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
        model = ItemItemRecommender(shrinkage=0.0).fit(build_interactions(toy_ratings), toy_catalog)
        scores = [score for _, score in model.similar_items("b1", k=5)]
        assert scores == sorted(scores, reverse=True)
