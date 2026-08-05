"""Metric tests against hand-computed values.

Every expected number below is worked out by hand in the test itself. A metric checked
against a second implementation of the same formula would agree with its own mistakes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from recommender.eval import catalog_coverage_at_k, hit_rate_at_k, novelty_at_k
from recommender.models.base import top_k_from_scores


def test_hit_rate_counts_a_hit_anywhere_in_the_list() -> None:
    recommended = np.array([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]], dtype=object)
    holdout = np.array(["c", "x", "g"])  # hit at rank 3, miss, hit at rank 1
    assert hit_rate_at_k(recommended, holdout) == pytest.approx(2 / 3)


def test_hit_rate_is_zero_when_nothing_matches() -> None:
    recommended = np.array([["a", "b"], ["c", "d"]], dtype=object)
    assert hit_rate_at_k(recommended, np.array(["z", "y"])) == 0.0


def test_coverage_counts_distinct_catalogue_items_only() -> None:
    """'a' appears twice (counts once); 'ghost' is not in the catalogue (counts never)."""
    recommended = np.array([["a", "b"], ["a", "ghost"]], dtype=object)
    coverage = catalog_coverage_at_k(recommended, {"a", "b", "c", "d"}, catalog_size=100)
    assert coverage == pytest.approx(2 / 100)


def test_coverage_ignores_padding() -> None:
    recommended = np.array([["a", None], [None, None]], dtype=object)
    assert catalog_coverage_at_k(recommended, {"a", "b"}, catalog_size=10) == pytest.approx(1 / 10)


def test_novelty_is_higher_for_rarer_items() -> None:
    popular = np.array([["hit"]], dtype=object)
    obscure = np.array([["rare"]], dtype=object)
    popularity = {"hit": 999, "rare": 0}
    kwargs = {"popularity": popularity, "total_interactions": 1000, "catalog_size": 24}
    assert novelty_at_k(obscure, **kwargs) > novelty_at_k(popular, **kwargs)


def test_novelty_matches_the_hand_computed_value() -> None:
    """-log2((count + 1) / (total + catalog_size)) averaged over both slots.

    'hit':  (999 + 1) / (1000 + 24) = 1000/1024 -> -log2 = 0.0342...
    'rare': (0 + 1)   / (1000 + 24) = 1/1024    -> -log2 = 10.0
    """
    recommended = np.array([["hit", "rare"]], dtype=object)
    expected = (-math.log2(1000 / 1024) + -math.log2(1 / 1024)) / 2
    got = novelty_at_k(recommended, {"hit": 999, "rare": 0}, total_interactions=1000, catalog_size=24)
    assert got == pytest.approx(expected)


def test_novelty_is_finite_for_never_interacted_items() -> None:
    """Content models recommend zero-interaction books; the metric must stay finite."""
    got = novelty_at_k(np.array([["unseen"]], dtype=object), {}, total_interactions=500, catalog_size=10)
    assert math.isfinite(got)


class TestTopK:
    def test_returns_best_first(self) -> None:
        scores = np.array([[0.1, 0.9, 0.5, 0.3]])
        assert top_k_from_scores(scores, k=3).tolist() == [[1, 2, 3]]

    def test_blocked_columns_are_never_returned(self) -> None:
        """A user's own train items must not come back as recommendations."""
        scores = np.array([[0.9, 0.8, 0.7, 0.6]])
        picked = top_k_from_scores(scores, k=2, blocked=[np.array([0, 1])])
        assert picked.tolist() == [[2, 3]]

    def test_pads_with_minus_one_when_candidates_run_out(self) -> None:
        scores = np.array([[0.5, -np.inf, -np.inf]])
        assert top_k_from_scores(scores, k=3).tolist() == [[0, -1, -1]]

    def test_ties_are_broken_deterministically(self) -> None:
        scores = np.array([[0.5, 0.5, 0.5, 0.5]])
        first = top_k_from_scores(scores, k=2)
        second = top_k_from_scores(scores, k=2)
        assert first.tolist() == second.tolist()
