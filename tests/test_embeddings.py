"""Embedding-layer tests with an injected fake encoder.

No test downloads a model or touches the network: tests stay offline and
deterministic. The encoder here is a deterministic hand-built map from title text to a
3-dimensional vector, so every expected neighbour below can be reasoned about directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import BookCrossing, build_interactions
from recommender.models.embeddings import EmbeddingRecommender

# Three axes: fantasy, regency, other. The toy catalogue maps onto them by hand.
TOPIC_VECTORS = {
    "the hobbit j.r.r. tolkien": [1.0, 0.0, 0.0],
    "the lord of the rings j.r.r. tolkien": [0.99, 0.1, 0.0],
    "the silmarillion j.r.r. tolkien": [0.95, 0.0, 0.1],
    "pride and prejudice jane austen": [0.0, 1.0, 0.0],
    "emma jane austen": [0.0, 0.98, 0.05],
    "der steppenwolf hermann hesse": [0.1, 0.1, 1.0],
    "cooking with fire anonymous": [0.0, 0.0, 0.9],
}


def fake_encoder(texts: list[str]) -> np.ndarray:
    """Look up known titles; anything else gets a stable hash-derived vector."""
    out = []
    for text in texts:
        if text in TOPIC_VECTORS:
            out.append(TOPIC_VECTORS[text])
        else:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            out.append(rng.random(3).tolist())
    return np.asarray(out, dtype=np.float32)


def _fit(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing, tmp_path=None) -> EmbeddingRecommender:
    model = EmbeddingRecommender(encoder=fake_encoder, cache_dir=tmp_path)
    return model.fit(build_interactions(toy_ratings), toy_catalog)


def test_neighbours_follow_the_embedding_space(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    assert {isbn for isbn, _ in model.similar_items("b1", k=2)} == {"b2", "b3"}
    assert model.similar_items("b4", k=1)[0][0] == "b5"


def test_vectors_are_l2_normalized(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    """Cosine is computed as a plain dot product, so normalization is not optional."""
    model = _fit(toy_ratings, toy_catalog)
    np.testing.assert_allclose(np.linalg.norm(model.vectors, axis=1), 1.0, rtol=1e-5)


def test_covers_every_catalogue_book_including_uninteracted_ones(
    toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
) -> None:
    model = _fit(toy_ratings, toy_catalog)
    assert len(model.item_ids) == len(toy_catalog.books)
    assert model.similar_items("b7", k=2) != []


def test_never_recommends_a_seen_item(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    model = _fit(toy_ratings, toy_catalog)
    for user in [1, 2, 3, 4, 5, 6]:
        seen = set(toy_ratings.loc[toy_ratings["User-ID"] == user, "ISBN"])
        picked = {isbn for isbn in model.recommend([user], k=6)[0].tolist() if isbn is not None}
        assert picked & seen == set()


def test_unknown_isbn_returns_nothing(toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
    assert _fit(toy_ratings, toy_catalog).similar_items("no-such-isbn") == []


class TestFuzzyLookup:
    """The app's input path: free text -> catalogue entry, no exact match needed."""

    def test_finds_the_book_from_an_inexact_query(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        model = _fit(toy_ratings, toy_catalog)
        # The encoder maps this query onto the fantasy axis, as a real multilingual
        # encoder would map "herr der ringe" onto The Lord of the Rings.
        TOPIC_VECTORS["tolkien ring book"] = [0.98, 0.05, 0.0]
        found = model.find_book("Tolkien Ring Book", k=3)
        assert {isbn for isbn, _ in found} <= {"b1", "b2", "b3"}

    def test_query_is_normalized_before_encoding(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        model = _fit(toy_ratings, toy_catalog)
        assert model.find_book("  THE HOBBIT J.R.R. TOLKIEN  ", k=1)[0][0] == "b1"

    def test_scores_are_ordered(self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
        model = _fit(toy_ratings, toy_catalog)
        scores = [score for _, score in model.find_book("emma jane austen", k=4)]
        assert scores == sorted(scores, reverse=True)


class TestCache:
    def test_vectors_are_written_and_reused(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing, tmp_path
    ) -> None:
        first = _fit(toy_ratings, toy_catalog, tmp_path=tmp_path)
        assert first.cache_path.exists()

        calls = {"n": 0}

        def counting_encoder(texts: list[str]) -> np.ndarray:
            calls["n"] += 1
            return fake_encoder(texts)

        second = EmbeddingRecommender(encoder=counting_encoder, cache_dir=tmp_path)
        second.fit(build_interactions(toy_ratings), toy_catalog)
        assert calls["n"] == 0, "second fit should have loaded the cached vectors"
        np.testing.assert_allclose(first.vectors, second.vectors)

    def test_cache_key_changes_when_the_catalogue_changes(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing, tmp_path
    ) -> None:
        first = _fit(toy_ratings, toy_catalog, tmp_path=tmp_path)
        bigger = toy_catalog.books.copy()
        bigger.loc[len(bigger)] = ["b8", "New Book", "New Author", "2020", "P", "s", "m", "l"]
        changed = BookCrossing(
            ratings=toy_catalog.ratings, books=bigger, users=toy_catalog.users, n_repaired=0
        )
        second = EmbeddingRecommender(encoder=fake_encoder, cache_dir=tmp_path)
        second.fit(build_interactions(toy_ratings), changed)
        assert second.cache_path != first.cache_path


def test_requires_fit_before_use(toy_catalog: BookCrossing) -> None:
    with pytest.raises(RuntimeError):
        EmbeddingRecommender(encoder=fake_encoder).similar_items("b1")


class TestGeometrySplit:
    """The two product paths use different geometry, on purpose (see find_book)."""

    def test_lookup_uses_uncentered_vectors(self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
        model = EmbeddingRecommender(encoder=fake_encoder, cache_dir=None, center=True)
        model.fit(build_interactions(toy_ratings), toy_catalog)
        # Centering changed the scoring vectors but left the lookup copy alone.
        assert not np.allclose(model.vectors, model.lookup_vectors)
        assert model.find_book("the hobbit j.r.r. tolkien", k=1)[0][0] == "b1"

    def test_uncentered_model_shares_one_array(self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing) -> None:
        model = EmbeddingRecommender(encoder=fake_encoder, cache_dir=None, center=False)
        model.fit(build_interactions(toy_ratings), toy_catalog)
        np.testing.assert_allclose(model.vectors, model.lookup_vectors)

    def test_centering_removes_the_shared_direction(
        self, toy_ratings: pd.DataFrame, toy_catalog: BookCrossing
    ) -> None:
        model = EmbeddingRecommender(encoder=fake_encoder, cache_dir=None, center=True)
        model.fit(build_interactions(toy_ratings), toy_catalog)
        assert abs(model.vectors.mean(axis=0)).max() < abs(model.lookup_vectors.mean(axis=0)).max()
