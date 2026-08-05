"""Regression tests for the shared fit dispatch.

These exist because of a real bug. The runner special-cased the explicit-only ablation
and ALS's confidence weights; the notebook did not, and silently reported the ablation as
a byte-identical copy of the binarized run — a wrong number that looked entirely
plausible. The dispatch now lives in one place, and these tests pin what it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data import BookCrossing, build_interactions
from recommender.models import fit_model
from recommender.models.als import ALSRecommender
from recommender.models.item_item import ItemItemRecommender
from recommender.models.popularity import PopularityRecommender


def test_explicit_ablation_is_fitted_on_a_different_matrix(
    toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
) -> None:
    """The bug: both models ended up on the binarized matrix, so the ablation was a no-op."""
    train = build_interactions(toy_ratings)
    binary = fit_model(ItemItemRecommender(shrinkage=0.0), train, toy_catalog, toy_catalog.ratings)
    explicit = fit_model(
        ItemItemRecommender(shrinkage=0.0, signal="explicit"), train, toy_catalog, toy_catalog.ratings
    )
    assert binary.train.matrix.nnz != explicit.train.matrix.nnz
    assert (binary.similarity != explicit.similarity).nnz > 0


def test_explicit_ablation_drops_exactly_the_ungraded_rows(
    toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
) -> None:
    train = build_interactions(toy_ratings)
    explicit = fit_model(
        ItemItemRecommender(signal="explicit"), train, toy_catalog, toy_catalog.ratings
    )
    assert explicit.train.matrix.nnz == int(toy_catalog.ratings["is_explicit"].sum())


def test_ablation_keeps_the_same_index_space(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    """Different matrix, identical coordinates -- otherwise the two rows are not comparable."""
    train = build_interactions(toy_ratings)
    explicit = fit_model(
        ItemItemRecommender(signal="explicit"), train, toy_catalog, toy_catalog.ratings
    )
    np.testing.assert_array_equal(explicit.train.item_ids, train.item_ids)
    np.testing.assert_array_equal(explicit.train.user_ids, train.user_ids)


def test_binary_models_are_fitted_on_the_matrix_they_were_given(
    toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
) -> None:
    train = build_interactions(toy_ratings)
    model = fit_model(PopularityRecommender(), train, toy_catalog, toy_catalog.ratings)
    assert model.train is train


def test_als_receives_the_confidence_weights(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    """Without the ratings frame ALS falls back to a flat 1 + alpha, which is a different
    model. The dispatch is what makes sure it does not."""
    train = build_interactions(toy_ratings)
    weighted = fit_model(
        ALSRecommender(factors=4, iterations=3, alpha=20.0), train, toy_catalog, toy_catalog.ratings
    )
    flat = ALSRecommender(factors=4, iterations=3, alpha=20.0).fit(train, toy_catalog)
    assert not np.allclose(weighted.item_factors, flat.item_factors)
