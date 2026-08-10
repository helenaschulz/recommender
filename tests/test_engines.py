"""The three neighbour engines M23 makes switchable, and the property that makes A safe.

The load-bearing test here is the last one: configuration A must be the code that shipped.
M23 turned one hard-wired dot product into a seam, and the whole milestone rests on the
claim that swapping the engine changes the list and nothing else — floors, dedup, evidence
counting and reason sentences are shared. If the default path drifted, every published demo
number and the eleven rehearsed anchors would drift with it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recommender.engines import (
    CONFIGURATIONS,
    DEFAULT_CONFIGURATION,
    AlsFactors,
    ItemItemCosine,
    ReciprocalRankFusion,
    TfidfText,
    build_configuration,
    build_source,
    configuration,
)


@pytest.fixture
def readers() -> sp.csr_matrix:
    """Four items, five readers. Item 0 and 1 share three readers; 2 shares one with 0."""
    dense = np.array(
        [
            [1, 1, 1, 1, 0],  # item 0 — four readers
            [1, 1, 1, 0, 0],  # item 1 — three, all shared with item 0
            [1, 0, 0, 0, 0],  # item 2 — one, shared with item 0
            [0, 0, 0, 0, 1],  # item 3 — one, shared with nobody
        ],
        dtype=np.float64,
    )
    return sp.csr_matrix(dense)


@pytest.fixture
def support(readers) -> np.ndarray:
    return np.asarray(readers.sum(axis=1)).ravel()


def eligible_all(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


class TestItemItemCosine:
    def test_the_shrunk_cosine_is_the_published_formula(self, readers, support) -> None:
        """co / (sqrt(support_i · support_j) + λ), the formula in models/item_item.py."""
        source = ItemItemCosine(readers, support, shrinkage=10.0)
        source._eligible = eligible_all(4)
        rows, scores = source.candidates(0, take=3)
        expected = 3 / (np.sqrt(4 * 3) + 10.0)  # item 1: three shared readers
        assert rows[0] == 1
        assert scores[0] == pytest.approx(expected)

    def test_an_item_sharing_nobody_scores_zero(self, readers, support) -> None:
        source = ItemItemCosine(readers, support)
        source._eligible = eligible_all(4)
        rows, scores = source.candidates(0, take=3)
        assert scores[list(rows).index(3)] == 0.0

    def test_the_anchor_is_never_its_own_neighbour(self, readers, support) -> None:
        source = ItemItemCosine(readers, support)
        source._eligible = eligible_all(4)
        rows, _ = source.candidates(0, take=3)
        assert 0 not in rows.tolist()

    def test_an_ineligible_item_cannot_be_a_candidate(self, readers, support) -> None:
        """The candidate floor is the engine's to set, and the source must honour it."""
        source = ItemItemCosine(readers, support)
        mask = eligible_all(4)
        mask[1] = False  # the best neighbour, excluded
        source._eligible = mask
        rows, _ = source.candidates(0, take=3)
        assert 1 not in rows.tolist()

    def test_co_occurrence_is_the_apps_own_co_reader_count(self, readers, support) -> None:
        """The number the source ranks by and the number the app prints must agree."""
        source = ItemItemCosine(readers, support)
        co = source.co_occurrence(0)
        assert co[1] == 3.0
        assert co[2] == 1.0
        assert co[3] == 0.0


class TestReciprocalRankFusion:
    def test_an_item_both_halves_rank_beats_one_ranked_first_by_either(self, readers, support) -> None:
        """RRF's whole rule, and it is the M22 function doing the ranking."""
        collaborative = ItemItemCosine(readers, support)
        content = ItemItemCosine(readers, support)  # a stand-in second opinion
        fusion = ReciprocalRankFusion(collaborative, content)
        fusion._eligible = eligible_all(4)
        rows, scores = fusion.candidates(0, take=3)
        assert rows[0] == 1, "both halves rank item 1 first"
        assert scores[0] == pytest.approx(1 / 61 + 1 / 61)

    def test_the_floor_reaches_both_halves(self, readers, support) -> None:
        fusion = ReciprocalRankFusion(ItemItemCosine(readers, support), ItemItemCosine(readers, support))
        mask = eligible_all(4)
        mask[1] = False
        fusion._eligible = mask
        rows, _ = fusion.candidates(0, take=3)
        assert 1 not in rows.tolist(), "a floor set on the fusion must reach the halves it fuses"

    def test_it_reports_no_similarity_because_it_has_none(self) -> None:
        """M23 decision 7: the screen prints something honest or nothing."""
        assert ReciprocalRankFusion.score_label == ""
        assert AlsFactors.score_label and ItemItemCosine.score_label


class TestTfidfText:
    def test_an_item_with_no_title_is_never_a_candidate(self) -> None:
        books = pd.DataFrame(
            {"Book-Title": ["dune", "dune messiah", None], "Book-Author": ["herbert", "herbert", None]},
            index=pd.Index(["a", "b", "c"], name="ISBN"),
        )
        source = TfidfText(np.array(["a", "b", "c"], dtype=object), books, min_df=1)
        source._eligible = eligible_all(3)
        rows, _ = source.candidates(0, take=2)
        assert "c" not in [["a", "b", "c"][r] for r in rows.tolist()]

    def test_an_anchor_with_no_text_answers_nothing_rather_than_noise(self) -> None:
        books = pd.DataFrame(
            {"Book-Title": ["dune", "dune messiah", None], "Book-Author": ["herbert", "herbert", None]},
            index=pd.Index(["a", "b", "c"], name="ISBN"),
        )
        source = TfidfText(np.array(["a", "b", "c"], dtype=object), books, min_df=1)
        source._eligible = eligible_all(3)
        rows, scores = source.candidates(2, take=2)
        assert rows.size == 0 and scores.size == 0


class TestBuildSource:
    def test_an_unknown_engine_is_an_error_rather_than_a_default(self) -> None:
        with pytest.raises(ValueError, match="unknown engine"):
            build_source("word2vec", object())

    def test_it_still_builds_all_three_because_three_are_published(self) -> None:
        """L87, L88 and L89 were measured through this function on three engines.

        The *registry* is two — C is dropped and does not come back — but a published line's
        command has to keep running, and ``scripts/audit_floor_band.py`` is L89's command.
        """
        from tests.test_demo import _assets

        assets = _assets()
        assert {build_source(name, assets).name for name in ("als", "item-item", "rrf")} == {
            "als",
            "item-item",
            "rrf",
        }


class TestConfigurations:
    """What the app offers, which is deliberately not what :func:`build_source` builds."""

    def test_there_are_two_and_c_is_not_one_of_them(self) -> None:
        """M23.10: C is dropped and does not come back. L85 ranks it first and L89 says why."""
        assert set(CONFIGURATIONS) == {"A", "B"}
        assert all(config.engine != "rrf" for config in CONFIGURATIONS.values())

    def test_the_default_is_a(self) -> None:
        """Decision 2. The rehearsed engine is what the demo opens on."""
        assert DEFAULT_CONFIGURATION == "A"
        assert CONFIGURATIONS[DEFAULT_CONFIGURATION].engine == "als"

    def test_every_configuration_labels_its_own_number(self) -> None:
        """Decision 6: a configuration cannot arrive without a label for what it prints."""
        for config in CONFIGURATIONS.values():
            assert config.score_label and config.label and config.short_label and config.blurb

    def test_the_two_labels_are_different_because_the_numbers_are(self) -> None:
        labels = [config.score_label for config in CONFIGURATIONS.values()]
        assert len(set(labels)) == len(labels)

    def test_the_picker_and_the_table_row_name_the_same_model(self) -> None:
        """The screen may not carry two names for one thing (review,10.08.2026).

        For one afternoon it did: the picker read *Shared readers* while the accuracy-table
        row it marked read *Item-based collaborative filtering*. The picker label now has to
        **start with** the model name the table prints, so the two cannot drift apart again —
        which is the whole reason the sidebar stopped hard-coding those rows.
        """
        for config in CONFIGURATIONS.values():
            model = config.short_label.split(" (")[0]
            assert config.label.startswith(model), f"{config.key}: {config.label!r} vs {model!r}"

    def test_the_two_table_rows_are_distinct(self) -> None:
        rows = [config.short_label for config in CONFIGURATIONS.values()]
        assert len(set(rows)) == len(rows)

    def test_an_unknown_key_is_an_error_rather_than_a_default(self) -> None:
        with pytest.raises(ValueError, match="unknown configuration"):
            configuration("C")

    def test_a_configuration_passes_through_itself(self) -> None:
        assert configuration(CONFIGURATIONS["B"]) is CONFIGURATIONS["B"]

    def test_a_answers_from_the_factors_and_never_looks_for_a_table(self) -> None:
        """Decision 7: A ships its own 155.6 MB matrix, so giving it a table means a rebuild."""
        from tests.test_demo import _assets

        assert CONFIGURATIONS["A"].table is False
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any fallback warning would fail this test
            assert isinstance(build_configuration("A", _assets()), AlsFactors)

    def test_b_falls_back_to_a_live_computation_and_says_so(self) -> None:
        """A missing table an hour before a demo must not take the demo down — loudly."""
        from tests.test_demo import _assets

        with pytest.warns(RuntimeWarning, match="live computation"):
            source = build_configuration("B", _assets(), table_dir=Path("/nonexistent"))
        assert isinstance(source, ItemItemCosine)

    def test_live_forces_the_computation_without_touching_the_disk(self) -> None:
        from tests.test_demo import _assets

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert isinstance(build_configuration("B", _assets(), live=True), ItemItemCosine)


class TestConfigurationAIsTheEngineThatShipped:
    """M23's load-bearing guard, and the reason the seam is safe to add at all.

    The eleven rehearsed anchors were additionally verified byte-identical against the real
    assets by hand (M23.8's check, run early): 110 slots, scores to twelve decimals, co-reader
    counts and reason sentences all unchanged. That comparison needs the gitignored 894 MB
    artefacts, so what is pinned *here* — offline and deterministic, per the test policy — is the
    property underneath it: the default path is the ALS dot product, ranked the same way.
    """

    def test_the_default_source_is_als(self) -> None:
        from tests.test_demo import _assets

        from recommender.demo import DemoEngine

        engine = DemoEngine(_assets())
        assert isinstance(engine.source, AlsFactors)

    def test_the_default_ranking_is_the_raw_factor_dot_product(self) -> None:
        """What `similar` returned before the seam existed, recomputed the obvious way."""
        from tests.test_demo import _assets

        from recommender.demo import DemoEngine

        assets = _assets()
        engine = DemoEngine(assets)
        got = engine.similar("hobbit|tolkien", k=3, tau=0)

        scores = np.asarray(assets.factors @ np.asarray(assets.factors[0]), dtype=np.float64)
        scores[assets.item_support < assets.similar_min_support] = -np.inf
        scores[0] = -np.inf
        expected = [str(assets.item_ids[i]) for i in np.argsort(-scores, kind="stable")[: len(got)]]
        assert [s.isbn for s in got] == expected

    def test_the_candidate_floor_is_the_engines_and_reaches_the_source(self) -> None:
        """Swapping the engine must not be able to move a floor as a side effect."""
        from tests.test_demo import _assets

        from recommender.demo import DemoEngine

        assets = _assets()
        source = ItemItemCosine(assets.readers, assets.item_support)
        engine = DemoEngine(assets, source=source)
        assert source._eligible.tolist() == (assets.item_support >= assets.similar_min_support).tolist()
        assert not source._eligible[4], "the support-3 work is below the L34 floor and cannot be a candidate"
        assert all(s.isbn != str(assets.item_ids[4]) for s in engine.similar("hobbit|tolkien", k=5, tau=0))
