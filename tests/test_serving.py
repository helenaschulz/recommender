"""Serving-layer tests: the work dedup, against a stub model with a scripted output.

The wrapper is tested against a stub rather than a real model on purpose. What is under
test is the collapsing rule — keep the best ISBN per work, never one the user already
has — and a stub makes every expected list computable by hand. Whether item-item ranks
well is a different question, measured in the ledger, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import BookCrossing, Interactions, build_interactions, cluster_works
from recommender.models.base import Recommender
from recommender.serving import WorkDeduped, duplicate_slot_rate

#: Two works with two editions each, plus two singletons.
EDITION_BOOKS = pd.DataFrame(
    {
        "ISBN": ["h1", "h2", "e1", "e2", "s1", "s2"],
        "Book-Title": [
            "The Hobbit",
            "The Hobbit (Collector's Edition)",
            "Emma",
            "Emma (Penguin Classics)",
            "Dune",
            "Solaris",
        ],
        "Book-Author": [
            "J. R. R. Tolkien",
            "J.R.R. Tolkien",
            "Jane Austen",
            "Jane Austen",
            "Frank Herbert",
            "Stanislaw Lem",
        ],
    }
)


class ScriptedRecommender(Recommender):
    """Returns a fixed candidate list for every user and anchor."""

    name = "scripted"

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order

    def fit(self, train: Interactions, catalog: BookCrossing) -> ScriptedRecommender:
        self.train = train
        return self

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        row = (self.order + [None] * k)[:k]
        return np.array([row for _ in user_ids], dtype=object)

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        return [(other, 1.0 - i / 100) for i, other in enumerate(self.order[:k])]


@pytest.fixture
def works():
    return cluster_works(EDITION_BOOKS)


@pytest.fixture
def train() -> Interactions:
    """User 1 owns h1 (an edition of The Hobbit). User 2 owns nothing relevant."""
    ratings = pd.DataFrame(
        {"User-ID": [1, 1, 2], "ISBN": ["h1", "s2", "s1"], "Book-Rating": [8, 5, 9]},
    )
    return build_interactions(ratings, item_ids=np.array(EDITION_BOOKS["ISBN"]))


def test_the_two_editions_collapse_to_the_better_ranked_one(works, train) -> None:
    inner = ScriptedRecommender(["e1", "e2", "s1", "s2"]).fit(train, None)
    model = WorkDeduped(inner, works, oversample=4)
    out = model.recommend(np.array([2]), k=3)
    # e2 is the same work as e1 and drops out; the higher-ranked edition survives.
    # s1 is user 2's own train item, so it drops too and s2 moves up.
    assert list(out[0]) == ["e1", "s2", None]


def test_dedup_blocks_a_work_the_user_already_owns_under_another_isbn(works, train) -> None:
    """The L31 failure mode: user 1 owns h1, so h2 must never be recommended."""
    inner = ScriptedRecommender(["h2", "e1", "s1"]).fit(train, None)
    model = WorkDeduped(inner, works, oversample=4)
    assert list(model.recommend(np.array([1]), k=3)[0]) == ["e1", "s1", None]


def test_without_dedup_the_duplicate_is_returned(works, train) -> None:
    """The wrapper is doing the work — the inner model happily returns the duplicate."""
    inner = ScriptedRecommender(["h2", "e1", "s1"]).fit(train, None)
    assert inner.recommend(np.array([1]), k=3)[0][0] == "h2"


def test_unfilled_slots_stay_none_rather_than_being_padded(works, train) -> None:
    """An honest short list beats a full one padded with editions of the same book."""
    inner = ScriptedRecommender(["e1", "e2"]).fit(train, None)
    model = WorkDeduped(inner, works, oversample=5)
    out = model.recommend(np.array([2]), k=4)
    assert list(out[0]) == ["e1", None, None, None]


def test_similar_items_drops_other_editions_of_the_anchor(works, train) -> None:
    """*The Da Vinci Code*'s top seven neighbours were seven Da Vinci Codes (L31)."""
    inner = ScriptedRecommender(["h2", "e1", "e2", "s1"]).fit(train, None)
    model = WorkDeduped(inner, works, oversample=4)
    assert [isbn for isbn, _ in model.similar_items("h1", k=3)] == ["e1", "s1"]


def test_similar_items_keeps_the_inner_scores(works, train) -> None:
    inner = ScriptedRecommender(["e1", "e2", "s1"]).fit(train, None)
    model = WorkDeduped(inner, works, oversample=4)
    assert model.similar_items("h1", k=2) == [("e1", 1.0), ("s1", 0.98)]


def test_collapse_and_recommend_agree(works, train) -> None:
    """`collapse` is what the analysis script reuses; it must not be a second code path."""
    inner = ScriptedRecommender(["h2", "e1", "e2", "s1", "s2"]).fit(train, None)
    model = WorkDeduped(inner, works, oversample=5)
    users = np.array([1, 2])
    raw = inner.recommend(users, k=5 * 5)
    assert (model.collapse(raw, users, 3) == model.recommend(users, k=3)).all()


def test_duplicate_slot_rate_counts_slots_and_users(works, train) -> None:
    """User 1 owns The Hobbit and gets h2 back (1 of 2 slots); user 2 gets nothing owned."""
    recommended = np.array([["h2", "e1"], ["e1", "e2"]], dtype=object)
    stats = duplicate_slot_rate(recommended, np.array([1, 2]), train, works)
    assert stats["duplicate_slot_share"] == pytest.approx(0.25)
    assert stats["affected_user_share"] == pytest.approx(0.5)
    assert stats["filled_slots"] == 4.0


def test_duplicate_slot_rate_ignores_empty_slots(works, train) -> None:
    recommended = np.array([["h2", None], [None, None]], dtype=object)
    stats = duplicate_slot_rate(recommended, np.array([1, 2]), train, works)
    assert stats["duplicate_slot_share"] == pytest.approx(1.0)
    assert stats["filled_slots"] == 1.0


def test_params_record_that_dedup_was_on(works, train) -> None:
    """Every ledger row has to be able to say whether it was measured with dedup."""
    model = WorkDeduped(ScriptedRecommender([]).fit(train, None), works, oversample=7)
    assert "work-dedup" in model.describe_params()
    assert "oversample=7" in model.describe_params()
