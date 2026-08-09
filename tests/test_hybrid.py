"""The three combination rules, each pinned on a case worked out by hand.

The rules are small enough that every expectation below can be read off the inputs, which
is the point: a fusion verified against another implementation of fusion proves nothing.

The property that matters most here is the one the milestone's escalation rule names — a
hybrid must not quietly change what the collaborative model does on its own — so the first
test is that a cascade with candidates to spare *is* item-item, list for list.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommender.data import build_interactions
from recommender.models.content_tfidf import TfidfRecommender
from recommender.models.hybrid import (
    CASCADE,
    FUSION,
    RRF,
    RULES,
    HybridRecommender,
    combine_user,
    combine_user_scored,
)
from recommender.models.item_item import ItemItemRecommender


def lists(cf: list[tuple[str, float]], ct: list[tuple[str, float]]):
    """Pack two ranked (id, score) lists the way ``recommend_scored`` returns them."""
    cf_ids = np.array([i for i, _ in cf], dtype=object)
    cf_scores = np.array([s for _, s in cf], dtype=np.float64)
    ct_ids = np.array([i for i, _ in ct], dtype=object)
    ct_scores = np.array([s for _, s in ct], dtype=np.float64)
    return cf_ids, cf_scores, ct_ids, ct_scores


class TestCascade:
    def test_a_full_collaborative_list_is_returned_unchanged(self) -> None:
        """The rule the ledger describes: content is a filler, never a voter."""
        cf = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        ct = [("x", 0.99), ("y", 0.98), ("z", 0.97)]
        assert combine_user(*lists(cf, ct), rule=CASCADE, k=3) == ["a", "b", "c"]

    def test_content_fills_only_the_slots_left_over(self) -> None:
        cf = [("a", 0.9)]
        ct = [("x", 0.5), ("y", 0.4)]
        assert combine_user(*lists(cf, ct), rule=CASCADE, k=3) == ["a", "x", "y"]

    def test_the_filler_never_duplicates_a_slot_already_taken(self) -> None:
        cf = [("a", 0.9)]
        ct = [("a", 0.5), ("x", 0.4)]
        assert combine_user(*lists(cf, ct), rule=CASCADE, k=2) == ["a", "x"]

    def test_the_support_floor_forfeits_a_thin_slot_to_the_content_model(self) -> None:
        cf = [("a", 0.9), ("thin", 0.8)]
        ct = [("x", 0.5)]
        support = {"a": 50, "thin": 2}
        picked = combine_user(*lists(cf, ct), rule=CASCADE, k=2, support=support, support_floor=5)
        assert picked == ["a", "x"], "the below-floor collaborative slot is the one that goes"

    def test_an_empty_collaborative_list_falls_through_entirely(self) -> None:
        """The 13.34% of users item-item cannot reach at all (L73's leftmost stratum)."""
        picked = combine_user(*lists([], [("x", 0.5), ("y", 0.4)]), rule=CASCADE, k=2)
        assert picked == ["x", "y"]


class TestReciprocalRankFusion:
    def test_an_item_ranked_by_both_beats_an_item_ranked_first_by_one(self) -> None:
        """1/61 + 1/62 = 0.03253 against 1/61 = 0.01639 — the agreement is the whole rule."""
        cf = [("both", 1.0), ("cf-only", 0.9)]
        ct = [("both", 0.1), ("ct-only", 0.09)]
        assert combine_user(*lists(cf, ct), rule=RRF, k=1) == ["both"]

    def test_the_raw_scores_are_ignored(self) -> None:
        """Two runs differing only in score, identical in rank, must rank identically."""
        big = [("a", 900.0), ("b", 800.0)]
        small = [("a", 0.9), ("b", 0.8)]
        content = [("b", 0.5), ("c", 0.4)]
        assert combine_user(*lists(big, content), rule=RRF, k=3) == combine_user(
            *lists(small, content), rule=RRF, k=3
        )


class TestScoreFusion:
    def test_alpha_one_is_the_collaborative_ranking(self) -> None:
        cf = [("a", 5.0), ("b", 3.0)]
        ct = [("z", 0.9), ("a", 0.1)]
        assert combine_user(*lists(cf, ct), rule=FUSION, k=2, alpha=1.0)[0] == "a"

    def test_alpha_zero_puts_the_content_models_best_candidate_first(self) -> None:
        cf = [("a", 5.0), ("b", 3.0)]
        ct = [("z", 0.9), ("a", 0.1)]
        assert combine_user(*lists(cf, ct), rule=FUSION, k=1, alpha=0.0) == ["z"]

    def test_normalization_is_per_user_and_per_model(self) -> None:
        """Item-item's sums are ~100x TF-IDF's cosines; after min-max both span [0, 1].

        Without the normalization the collaborative side would win every slot at any alpha
        above about 0.01, which would make the alpha curve meaningless rather than flat.
        """
        cf = [("a", 500.0), ("b", 100.0)]
        ct = [("c", 0.9), ("d", 0.1)]
        # alpha 0.5: a -> 0.5, b -> 0.0, c -> 0.5, d -> 0.0. a and c tie and the
        # collaborative order breaks it; b and d tie the same way.
        assert combine_user(*lists(cf, ct), rule=FUSION, k=4, alpha=0.5) == ["a", "c", "b", "d"]

    def test_a_flat_candidate_list_normalizes_to_all_ones(self) -> None:
        """Every candidate equally good *for this user* is 1.0, not 0.0 — the alternative
        would hand the whole list to the other model, which is a rule nobody chose."""
        cf = [("a", 2.0), ("b", 2.0)]
        ct = [("c", 0.9), ("d", 0.1)]
        picked = combine_user(*lists(cf, ct), rule=FUSION, k=4, alpha=0.5)
        assert picked[:2] == ["a", "b"], "both collaborative candidates score 0.5, the content best 0.5"

    def test_an_unknown_rule_is_an_error_rather_than_a_default(self) -> None:
        with pytest.raises(ValueError, match="unknown rule"):
            combine_user(*lists([("a", 1.0)], []), rule="mean", k=1)


class TestAgainstTheBaseModels:
    """The escalation rule from M19, as a test: the cascade must not move item-item."""

    def test_a_cascade_with_full_lists_reproduces_item_item_exactly(self, toy_catalog) -> None:
        train = build_interactions(toy_catalog.ratings, weights="binary")
        collaborative = ItemItemRecommender(shrinkage=1.0, top_k_neighbours=10).fit(train, toy_catalog)
        content = TfidfRecommender(min_df=1).fit(train, toy_catalog)
        users = toy_catalog.ratings["User-ID"].unique()

        hybrid = HybridRecommender(collaborative, content, rule=CASCADE, candidates=10)
        base = collaborative.recommend(users, k=2)
        mixed = hybrid.recommend(users, k=2)
        for row in range(len(users)):
            filled = [i for i in base[row].tolist() if i is not None]
            if len(filled) == 2:
                assert list(mixed[row]) == filled, "a full collaborative list must survive the cascade"

    def test_recommend_is_a_slice_of_recommend_scored(self, toy_catalog) -> None:
        """One ranking path, so a hybrid cannot be fed a list the table never scored."""
        train = build_interactions(toy_catalog.ratings, weights="binary")
        users = toy_catalog.ratings["User-ID"].unique()
        for model in (
            ItemItemRecommender(shrinkage=1.0, top_k_neighbours=10).fit(train, toy_catalog),
            TfidfRecommender(min_df=1).fit(train, toy_catalog),
        ):
            ids, scores = model.recommend_scored(users, k=3)
            assert np.array_equal(model.recommend(users, k=3), ids)
            for row in range(len(users)):
                usable = [s for s in scores[row].tolist() if s > -np.inf]
                assert usable == sorted(usable, reverse=True), "scores must come back ranked"


class TestTheItemToItemPath:
    """``similar_items`` under all three rules — the surface M22 is the first to measure.

    Until M22 only the cascade was ever built item-to-item (``measure_anchor_hitrate.py``
    hard-wired it), so the fused rules reached this method in no test and no measurement.
    """

    def test_the_reported_score_is_the_one_the_rule_ranked_by(self) -> None:
        """RRF's own contract: an item both models rank wins, and its number says why.

        Before M22 this returned the *base model's* score, so the list came back
        ``[("b", 0.1), ("a", 0.9)]`` — sorted descending by a column printed ascending. That
        is the M17 finding one layer down, and it is why the number and the ranking now come
        out of the same function.
        """
        cf = [("a", 0.9), ("b", 0.1)]
        ct = [("b", 0.5), ("c", 0.4)]
        picked = combine_user_scored(*lists(cf, ct), rule=RRF, k=3)
        assert [i for i, _ in picked] == ["b", "a", "c"]
        # b is second in the collaborative list and first in the content one: 1/62 + 1/61.
        assert picked[0][1] == pytest.approx(1 / 62 + 1 / 61), "b is ranked first because both models list it"
        assert [s for _, s in picked] == sorted((s for _, s in picked), reverse=True)

    def test_a_cascade_reports_the_contributing_models_own_score(self) -> None:
        """Nothing is fused under a cascade, so a fused-looking number would be an invention."""
        cf = [("a", 0.9)]
        ct = [("x", 0.42)]
        assert combine_user_scored(*lists(cf, ct), rule=CASCADE, k=2) == [("a", 0.9), ("x", 0.42)]

    def test_combine_user_is_combine_user_scored_with_the_scores_dropped(self) -> None:
        """One ranking path: the ids cannot depend on whether the caller wanted scores."""
        cf = [("a", 5.0), ("b", 3.0), ("c", 1.0)]
        ct = [("c", 0.9), ("d", 0.5), ("a", 0.1)]
        for rule in RULES:
            scored = combine_user_scored(*lists(cf, ct), rule=rule, k=3, alpha=0.6)
            plain = combine_user(*lists(cf, ct), rule=rule, k=3, alpha=0.6)
            assert [i for i, _ in scored] == plain, rule

    def test_every_rule_returns_a_ranked_list_of_neighbours(self, toy_catalog) -> None:
        train = build_interactions(toy_catalog.ratings, weights="binary")
        collaborative = ItemItemRecommender(shrinkage=1.0, top_k_neighbours=10).fit(train, toy_catalog)
        content = TfidfRecommender(min_df=1).fit(train, toy_catalog)
        anchor = toy_catalog.ratings["ISBN"].iloc[0]

        for rule in RULES:
            hybrid = HybridRecommender(collaborative, content, rule=rule, alpha=0.6, candidates=10)
            neighbours = hybrid.similar_items(anchor, k=3)
            ids = [i for i, _ in neighbours]
            scores = [s for _, s in neighbours]
            assert anchor not in ids, f"{rule} returned the anchor as its own neighbour"
            assert len(ids) == len(set(ids)), f"{rule} returned a duplicate slot"
            assert scores == sorted(scores, reverse=True), f"{rule} reported scores out of rank order"

    def test_an_anchor_neither_model_knows_returns_nothing_rather_than_padding(self, toy_catalog) -> None:
        train = build_interactions(toy_catalog.ratings, weights="binary")
        collaborative = ItemItemRecommender(shrinkage=1.0, top_k_neighbours=10).fit(train, toy_catalog)
        content = TfidfRecommender(min_df=1).fit(train, toy_catalog)
        hybrid = HybridRecommender(collaborative, content, rule=RRF, candidates=10)
        assert hybrid.similar_items("no-such-isbn", k=3) == []

    def test_alpha_reaches_the_item_query(self) -> None:
        """M22 inherits alpha=0.6 rather than re-tuning it, so it has to actually arrive."""
        cf = [("a", 5.0), ("b", 3.0)]
        ct = [("z", 0.9), ("a", 0.1)]
        assert combine_user(*lists(cf, ct), rule=FUSION, k=1, alpha=1.0) == ["a"]
        assert combine_user(*lists(cf, ct), rule=FUSION, k=1, alpha=0.0) == ["z"]
