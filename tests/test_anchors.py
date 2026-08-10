"""The item-to-item metric: the anchor draw and the neighbour matrix it feeds.

Two modules meet here — the draw lives in :mod:`recommender.split` next to the holdout draw,
the scoring in :mod:`recommender.eval` next to the other metrics — so the tests live in one
place named for the behaviour rather than split across two named for the modules.

What these pin down is the *pairing*. AnchorHitRate@10 is only comparable to HitRate@10
because both are scored on the same readers, in the same row order, with the same definition
of a hit. Nothing in the code raises when that alignment breaks; McNemar quietly tests noise
instead. So the alignment is a test, not a comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import build_interactions
from recommender.eval import evaluate_anchors, neighbour_matrix, owned_items
from recommender.models.base import Recommender
from recommender.split import make_split, pick_anchors


class StubNeighbours(Recommender):
    """A model whose neighbourhoods are whatever the test says they are."""

    name = "stub"

    def __init__(self, table: dict[str, list[str]]) -> None:
        super().__init__()
        self.table = table
        self.params = {"stub": True}
        self.asked: list[tuple[str, int]] = []

    def fit(self, train, catalog):  # pragma: no cover - never fitted in these tests
        self.train = train
        return self

    def recommend(self, user_ids, k=10):  # pragma: no cover - not the surface under test
        raise NotImplementedError

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        self.asked.append((isbn, k))
        return [(item, 1.0 / (rank + 1)) for rank, item in enumerate(self.table.get(isbn, []))][:k]


# --- the draw ---------------------------------------------------------------------------


def test_one_anchor_per_eligible_user_in_test_row_order(toy_ratings: pd.DataFrame) -> None:
    """The load-bearing property: anchors line up with split.test element for element."""
    split = make_split(toy_ratings)
    anchors = pick_anchors(split)
    assert np.array_equal(anchors.user_ids, split.test["User-ID"].to_numpy())
    assert len(anchors.frame) == len(split.test) == 3


def test_anchor_is_never_the_held_out_item(toy_ratings: pd.DataFrame) -> None:
    """The anchor comes from train, so it cannot be the target it is meant to find."""
    split = make_split(toy_ratings)
    anchors = pick_anchors(split)
    assert not (anchors.item_ids == split.test["ISBN"].to_numpy()).any()


def test_anchor_is_a_train_row_of_that_same_reader(toy_ratings: pd.DataFrame) -> None:
    split = make_split(toy_ratings)
    anchors = pick_anchors(split)
    train_pairs = set(zip(split.train["User-ID"], split.train["ISBN"], strict=True))
    assert all(pair in train_pairs for pair in zip(anchors.user_ids, anchors.item_ids, strict=True))


def test_the_relevant_pool_is_preferred_and_the_fallback_is_flagged(toy_ratings: pd.DataFrame) -> None:
    """User 4 grades exactly one book >=8, so the holdout takes it and the fallback fires.

    Users 1 and 6 keep at least one rating >=8 in train, so theirs come from the preferred
    pool at a grade >= 8. That split is worked out from the fixture by hand, not read off a
    run, which is what makes it a test.
    """
    split = make_split(toy_ratings)
    anchors = pick_anchors(split)
    by_user = anchors.frame.set_index("User-ID")

    assert bool(by_user.loc[4, "fallback"]) is True
    assert by_user.loc[4, "Book-Rating"] < split.relevance_threshold
    for user in (1, 6):
        assert bool(by_user.loc[user, "fallback"]) is False
        assert by_user.loc[user, "Book-Rating"] >= split.relevance_threshold
    assert anchors.n_fallback == 1


def test_the_draw_is_seed_stable_and_input_order_independent(toy_ratings: pd.DataFrame) -> None:
    """Same seed, same anchors — even when the rows arrive shuffled."""
    split = make_split(toy_ratings)
    shuffled = make_split(toy_ratings.sample(frac=1.0, random_state=7))

    assert np.array_equal(pick_anchors(split).item_ids, pick_anchors(split).item_ids)
    assert np.array_equal(
        pick_anchors(split).item_ids,
        pick_anchors(shuffled).item_ids[np.argsort(shuffled.test["User-ID"].to_numpy())],
    )


def test_a_different_seed_is_allowed_to_move_the_draw(toy_ratings: pd.DataFrame) -> None:
    """The anchor rule is a free parameter, so the seed has to be able to matter."""
    split = make_split(toy_ratings)
    seeds = {tuple(pick_anchors(split, seed=seed).item_ids) for seed in range(12)}
    assert len(seeds) > 1


def test_missing_anchor_is_an_error_not_a_silent_miss() -> None:
    """A Split whose test user has no graded train row must raise rather than score 0."""
    ratings = pd.DataFrame(
        {"User-ID": [1, 1, 1, 1, 1], "ISBN": ["b1", "b2", "b3", "b4", "b5"], "Book-Rating": [9, 8, 10, 7, 6]}
    )
    split = make_split(ratings)
    starved = split.__class__(
        train=split.train.iloc[0:0],
        test=split.test,
        seed=split.seed,
        min_explicit=split.min_explicit,
        relevance_threshold=split.relevance_threshold,
    )
    with pytest.raises(ValueError, match="no graded train row"):
        pick_anchors(starved)


# --- the neighbour matrix ---------------------------------------------------------------


def test_owned_items_are_filtered_out_of_the_neighbour_list() -> None:
    """The profile column excludes what the reader has read; this column must too."""
    model = StubNeighbours({"a": ["own1", "keep1", "own2", "keep2"]})
    neighbours, answered = neighbour_matrix(model, np.array(["a"]), k=2, owned=[frozenset({"own1", "own2"})])
    assert list(neighbours[0]) == ["keep1", "keep2"]
    assert answered.tolist() == [True]


def test_depth_requested_covers_the_filter() -> None:
    """k alone would leave fewer than k slots once the reader's shelf is removed."""
    model = StubNeighbours({"a": []})
    neighbour_matrix(model, np.array(["a"]), k=10, owned=[frozenset({"x", "y", "z"})])
    assert model.asked == [("a", 13)]


def test_an_unanswered_anchor_is_a_padded_miss_not_a_dropped_reader() -> None:
    """A model must not improve its rate by declining to answer."""
    model = StubNeighbours({"a": ["hit"]})
    neighbours, answered = neighbour_matrix(model, np.array(["a", "unknown"]), k=3)
    assert list(neighbours[1]) == [None, None, None]
    assert answered.tolist() == [True, False]


def test_answered_is_taken_before_filtering_not_after() -> None:
    """"No representation for that book" and "everything it knew, you had read" differ."""
    model = StubNeighbours({"a": ["own"]})
    _, answered = neighbour_matrix(model, np.array(["a"]), k=3, owned=[frozenset({"own"})])
    assert answered.tolist() == [True]


def test_padding_never_scores_a_hit() -> None:
    """None must not match a held-out item, however the comparison is broadcast."""
    ratings = pd.DataFrame(
        {
            "User-ID": [1] * 6 + [2] * 6,
            "ISBN": ["b1", "b2", "b3", "b4", "b5", "b6"] * 2,
            "Book-Rating": [9, 8, 10, 7, 6, 5, 9, 8, 10, 7, 6, 5],
        }
    )
    split = make_split(ratings)
    train = build_interactions(split.train, weights="binary")
    anchors = pick_anchors(split)
    model = StubNeighbours({})  # knows nothing about any anchor

    result = evaluate_anchors(
        model,
        split,
        train,
        anchors.item_ids,
        catalog_isbns=set(train.item_ids.tolist()),
        catalog_size=len(train.item_ids),
        k=5,
    )
    assert result.anchor_hit_rate == 0.0
    assert result.answered == 0.0


def test_a_broken_pairing_raises_rather_than_scoring_noise() -> None:
    ratings = pd.DataFrame(
        {"User-ID": [1] * 6, "ISBN": ["b1", "b2", "b3", "b4", "b5", "b6"], "Book-Rating": [9, 8, 10, 7, 6, 5]}
    )
    split = make_split(ratings)
    train = build_interactions(split.train, weights="binary")
    with pytest.raises(ValueError, match="pairing is broken"):
        evaluate_anchors(
            StubNeighbours({}),
            split,
            train,
            np.array(["b1", "b2"]),  # two anchors, one held-out item
            catalog_isbns=set(train.item_ids.tolist()),
            catalog_size=len(train.item_ids),
        )


def test_owned_items_reads_the_readers_own_train_row(toy_ratings: pd.DataFrame) -> None:
    split = make_split(toy_ratings)
    train = build_interactions(split.train, weights="binary")
    owned = owned_items(train, split.test["User-ID"].to_numpy())
    user_1_train = set(split.train.loc[split.train["User-ID"] == 1, "ISBN"])
    assert owned[0] == user_1_train
    assert split.test.loc[0, "ISBN"] not in owned[0]


def test_the_hit_is_the_shared_definition(toy_ratings: pd.DataFrame) -> None:
    """A model that returns exactly the held-out book scores 1.0 — no second hit rule."""
    split = make_split(toy_ratings)
    train = build_interactions(split.train, weights="binary")
    anchors = pick_anchors(split)
    # Accumulated rather than assigned: two readers can legitimately draw the same anchor,
    # and a dict comprehension would drop the first of them and fail for the wrong reason.
    table: dict[str, list[str]] = {}
    for anchor, held in zip(anchors.item_ids, split.test["ISBN"], strict=True):
        table.setdefault(anchor, []).append(held)

    result = evaluate_anchors(
        StubNeighbours(table),
        split,
        train,
        anchors.item_ids,
        catalog_isbns=set(train.item_ids.tolist()),
        catalog_size=len(train.item_ids),
        k=10,
        owned=owned_items(train, anchors.user_ids),
    )
    assert result.anchor_hit_rate == 1.0
    assert result.answered == 1.0
