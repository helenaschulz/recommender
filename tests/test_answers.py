"""The precomputed answer table (M23.10.1): its round trip, and its refusal to answer stale.

A serving cache that answers with the right numbers attached to the wrong books is the worst
failure available to this layer, and it is silent — a recommendation list has no test that
fails. So the load-bearing tests here are the staleness ones: every field of the stamp is
proved to stop a load, including the content hash, which is the only one that can see a
rebuild that kept the shape.

Offline and deterministic, per the test policy: the table is written from hand-built arrays into
``tmp_path`` and read straight back. No data files and no assets.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.test_demo import DUNE, EMMA, HOBBIT, RING, UBIK, _assets

from recommender.answers import (
    TABLE_VERSION,
    AnswerTable,
    askable_rows,
    assets_stamp,
    load_answers,
    save_answers,
    table_path,
)


def _write(directory, assets, *, engine="item-item", anchors=(0, 1), rows=(1, 2, 0, 3), indptr=(0, 2, 4)):
    return save_answers(
        table_path(directory, engine),
        engine=engine,
        score_label="shared readers",
        anchors=np.array(anchors, dtype=np.int32),
        indptr=np.array(indptr, dtype=np.int32),
        rows=np.array(rows, dtype=np.int32),
        scores=np.arange(len(rows), dtype=np.float32) / 10.0,
        assets=assets,
        meta={"k": 2, "built_at": "2026-08-10"},
    )


class TestAskableRows:
    def test_it_is_the_floor_and_the_catalogue_together(self) -> None:
        """L65's filter: above the anchor floor **and** nameable. Ubik fails the first."""
        assets = _assets()
        assert [str(assets.item_ids[r]) for r in askable_rows(assets)] == [HOBBIT, RING, EMMA, DUNE]

    def test_a_row_the_catalogue_cannot_name_is_not_askable(self) -> None:
        """The 24 real ones (L46): above the floor, no catalogue row, so never an anchor."""
        assets = _assets(books=_assets().books.drop(index=[DUNE]))
        assert DUNE not in [str(assets.item_ids[r]) for r in askable_rows(assets)]


class TestRoundTrip:
    def test_what_was_written_comes_back(self, tmp_path) -> None:
        assets = _assets()
        _write(tmp_path, assets)
        table = load_answers(tmp_path, "item-item", assets)
        rows, scores = table.candidates(0, take=10)
        assert rows.tolist() == [1, 2]
        assert scores.tolist() == pytest.approx([0.0, 0.1])

    def test_an_anchor_the_table_does_not_cover_answers_nothing(self, tmp_path) -> None:
        """Not a fallback to a live computation: the table is the answer or there is none."""
        assets = _assets()
        _write(tmp_path, assets)
        rows, scores = load_answers(tmp_path, "item-item", assets).candidates(4, take=10)
        assert rows.size == 0 and scores.size == 0

    def test_take_is_honoured_even_though_the_depth_was_spent_at_build_time(self, tmp_path) -> None:
        assets = _assets()
        _write(tmp_path, assets)
        rows, _ = load_answers(tmp_path, "item-item", assets).candidates(0, take=1)
        assert rows.tolist() == [1]

    def test_the_candidate_floor_still_reaches_it(self, tmp_path) -> None:
        """M23 decision 6: the floor is the engine's decision, so a table must honour it too.

        A source that ignored the mask would make a table the one configuration where
        switching the engine could move a floor as a side effect.
        """
        assets = _assets()
        _write(tmp_path, assets)
        table = load_answers(tmp_path, "item-item", assets)
        mask = np.ones(len(assets.item_ids), dtype=bool)
        mask[1] = False
        table._eligible = mask
        rows, scores = table.candidates(0, take=10)
        assert rows.tolist() == [2] and len(scores) == 1

    def test_it_is_a_neighbour_source_the_engine_cannot_tell_apart(self, tmp_path) -> None:
        """The property the whole milestone rests on: same dedup, same evidence, same reasons."""
        from recommender.demo import DemoEngine

        assets = _assets()
        _write(tmp_path, assets)
        engine = DemoEngine(assets, source=load_answers(tmp_path, "item-item", assets))
        got = engine.similar(HOBBIT, k=10)
        assert [s.isbn for s in got] == [RING, EMMA]
        assert got[0].reason and got[0].evidence.anchor_readers == 2


class TestStaleness:
    """Every field of the stamp must be able to stop a load, or it is decoration."""

    def test_a_missing_table_names_the_command_that_builds_it(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="build_answer_table"):
            load_answers(tmp_path, "item-item", _assets())

    def test_a_different_item_count_is_refused(self, tmp_path) -> None:
        assets = _assets()
        _write(tmp_path, assets)
        grown = replace(
            assets,
            item_ids=np.append(assets.item_ids, "new|author"),
            item_support=np.append(assets.item_support, 99),
        )
        with pytest.raises(RuntimeError, match="n_items"):
            load_answers(tmp_path, "item-item", grown)

    def test_a_different_work_key_is_refused(self, tmp_path) -> None:
        """L64 moved 0.5% of works and with them a third of every neighbourhood (L67)."""
        assets = _assets()
        _write(tmp_path, assets)
        with pytest.raises(RuntimeError, match="work_key"):
            load_answers(tmp_path, "item-item", replace(assets, work_key="something else"))

    def test_a_moved_floor_is_refused(self, tmp_path) -> None:
        assets = _assets()
        _write(tmp_path, assets)
        with pytest.raises(RuntimeError, match="anchor_min_support"):
            load_answers(tmp_path, "item-item", replace(assets, anchor_min_support=99))

    def test_a_rebuild_that_kept_the_shape_is_refused(self, tmp_path) -> None:
        """The one the other three cannot see, and the reason the stamp carries a hash."""
        assets = _assets()
        _write(tmp_path, assets)
        moved = replace(assets, item_support=np.array([50, 50, 50, 60, 3], dtype=np.int64))
        with pytest.raises(RuntimeError, match="item_support_sha256"):
            load_answers(tmp_path, "item-item", moved)

    def test_an_old_layout_is_refused_by_version(self, tmp_path) -> None:
        assets = _assets()
        path = _write(tmp_path, assets)
        with np.load(path, allow_pickle=True) as handle:
            payload = {name: handle[name] for name in handle.files}
        import json

        meta = json.loads(str(payload["meta"].item()))
        meta["table_version"] = TABLE_VERSION + 1
        payload["meta"] = np.array(json.dumps(meta), dtype=object)
        np.savez(path, **payload)
        with pytest.raises(RuntimeError, match="table version"):
            load_answers(tmp_path, "item-item", assets)

    def test_the_stamp_is_stable_for_unchanged_assets(self) -> None:
        assert assets_stamp(_assets()) == assets_stamp(_assets())


class TestProvenance:
    def test_it_says_where_it_came_from(self, tmp_path) -> None:
        assets = _assets()
        _write(tmp_path, assets)
        assert "2026-08-10" in load_answers(tmp_path, "item-item", assets).provenance

    def test_the_score_label_travels_with_the_table(self, tmp_path) -> None:
        """M23 decision 6: a configuration cannot arrive without a label for its number."""
        assets = _assets()
        _write(tmp_path, assets)
        assert load_answers(tmp_path, "item-item", assets).score_label == "shared readers"

    def test_the_name_is_the_engine_it_was_built_from(self) -> None:
        table = AnswerTable(engine="item-item", score_label="", index={}, rows=np.array([]),
                            scores=np.array([]), meta={})
        assert table.name == "item-item"


def test_ubik_is_the_below_floor_fixture_this_module_relies_on() -> None:
    """Guard: several tests above read as nonsense if the shared fixture stops matching."""
    assets = _assets()
    assert str(assets.item_ids[4]) == UBIK
    assert assets.item_support[4] < assets.anchor_floor
