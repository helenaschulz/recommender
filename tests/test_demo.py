"""Demo-engine tests: the reason sentences and the query rules, against hand-built assets.

The Streamlit layer is deliberately untested — it is widgets. Everything that could be
*wrong* rather than ugly lives in :mod:`recommender.demo` and is exercised here, offline,
with no data files, no model download and no network.

The fixture is keyed by **work**, like the real assets: editions are merged by
``scripts/build_app_assets.py`` before anything the engine sees, so "two ISBNs of one book"
is not a state this module can be in.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recommender.demo import (
    ASSET_VERSION,
    DemoAssets,
    DemoEngine,
    Evidence,
    load_assets,
    reason_sentence,
)

HOBBIT, RING, EMMA, DUNE, UBIK = (
    "the hobbit|tolkien",
    "the fellowship of the ring|tolkien",
    "emma|austen",
    "dune|herbert",
    "ubik|dick",
)
ITEMS = np.array([HOBBIT, RING, EMMA, DUNE, UBIK], dtype=object)

BOOKS = pd.DataFrame(
    {
        "ISBN": ITEMS,
        "Book-Title": ["The Hobbit", "The Fellowship of the Ring", "Emma", "Dune", "Ubik"],
        "Book-Author": ["Tolkien", "Tolkien", "Austen", "Herbert", "Dick"],
        "Year-Of-Publication": ["1937", "1954", "1815", "1965", "1969"],
        "series": ["Middle-earth", "Middle-earth", "", "", ""],
        "Image-URL-M": ["u1", "u2", "u3", "u4", "u5"],
    }
).set_index("ISBN")


def _assets(*, support=(50, 50, 50, 50, 3), readers=None, books=BOOKS) -> DemoAssets:
    """Five works in two dimensions, so every similarity below is checkable by hand.

    Hobbit at 0 degrees, Fellowship at 4, Emma at 30, Dune at 90, Ubik at 180 — Emma is
    deliberately outside LOOKUP_TIE_MARGIN of the other two.
    """
    angles = np.deg2rad([0.0, 4.0, 30.0, 90.0, 180.0])
    factors = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    if readers is None:
        # user 0 read Hobbit and Emma; user 1 read Hobbit and Dune; user 2 read Fellowship.
        readers = sp.csr_matrix(
            np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        )
    return DemoAssets(
        factors=factors,
        item_ids=ITEMS,
        item_support=np.array(support, dtype=np.int64),
        readers=readers,
        lookup_vectors=factors,
        lookup_ids=ITEMS,
        lookup_support=np.array(support, dtype=np.int64),
        books=books,
        similar_min_support=20,
        encoder_model="stub",
    )


class TestReasonSentence:
    """Pure, deterministic, and the only thing the app ever says about *why*."""

    def test_shared_readers_lead(self) -> None:
        got = reason_sentence(Evidence(score=0.42, co_readers=27, anchor_readers=806, same_author=False), "Dune")
        assert got == "27 readers of *Dune* also read this (similarity 0.42)."

    def test_one_reader_is_singular(self) -> None:
        got = reason_sentence(Evidence(score=0.5, co_readers=1, anchor_readers=9, same_author=False), "Dune")
        assert got.startswith("1 reader of *Dune* also read this")

    def test_metadata_clauses_follow_the_count(self) -> None:
        got = reason_sentence(
            Evidence(score=0.9, co_readers=4, anchor_readers=10, same_author=True, shared_series="Middle-earth"),
            "The Hobbit",
        )
        assert got == (
            "4 readers of *The Hobbit* also read this — same author, same series (Middle-earth) (similarity 0.90)."
        )

    def test_metadata_alone_still_makes_a_sentence(self) -> None:
        got = reason_sentence(Evidence(score=0.7, co_readers=0, anchor_readers=10, same_author=True), "Emma")
        assert got == "Same author (similarity 0.70)."

    def test_no_evidence_says_so_instead_of_inventing_some(self) -> None:
        got = reason_sentence(Evidence(score=0.31, co_readers=0, anchor_readers=10, same_author=False), "Ubik")
        assert got == "Close in the model's neighbourhood of *Ubik* (similarity 0.31)."

    def test_it_is_deterministic(self) -> None:
        evidence = Evidence(score=0.5, co_readers=2, anchor_readers=8, same_author=True)
        assert reason_sentence(evidence, "Dune") == reason_sentence(evidence, "Dune")


class TestSimilar:
    def test_the_nearest_work_comes_first(self) -> None:
        engine = DemoEngine(_assets(support=(50, 50, 50, 50, 50)))
        assert [s.isbn for s in engine.similar(HOBBIT, k=3)] == [RING, EMMA, DUNE]

    def test_the_anchor_never_recommends_itself(self) -> None:
        engine = DemoEngine(_assets())
        assert HOBBIT not in [s.isbn for s in engine.similar(HOBBIT, k=4)]

    def test_low_support_candidates_are_filtered_out(self) -> None:
        """Ledger L34: below the floor an item's factor is a noise direction, and with
        enough of them one will align with anything by chance."""
        engine = DemoEngine(_assets(support=(50, 50, 50, 50, 3)))
        assert UBIK not in [s.isbn for s in engine.similar(HOBBIT, k=5)]

    def test_a_low_support_anchor_is_refused_rather_than_answered(self) -> None:
        """Stricter than ALSRecommender.similar_items on purpose: the L34 argument applies
        to the vector being *queried* as much as to the ones being ranked, and a visitor
        can type anything."""
        engine = DemoEngine(_assets(support=(50, 50, 50, 50, 3)))
        assert engine.similar(UBIK) == []

    def test_an_unknown_book_returns_nothing_rather_than_something(self) -> None:
        engine = DemoEngine(_assets())
        assert engine.similar("not-a-work") == []

    def test_a_book_we_cannot_name_is_never_shown(self) -> None:
        """10.3% of interactions point at ISBNs with no catalogue row (L14). A card
        reading "[unknown 0432534220]" is worse than one fewer suggestion (L46)."""
        engine = DemoEngine(_assets(support=(50, 50, 50, 50, 50), books=BOOKS.drop(index=[RING])))
        assert [s.isbn for s in engine.similar(HOBBIT, k=3)] == [EMMA, DUNE, UBIK]

    def test_co_reader_counts_come_from_the_matrix(self) -> None:
        """Hobbit is read by users 0 and 2; Emma by user 0 -> exactly one shared reader."""
        engine = DemoEngine(_assets(support=(50, 50, 50, 50, 50)))
        by_id = {s.isbn: s for s in engine.similar(HOBBIT, k=4)}
        assert by_id[EMMA].evidence.co_readers == 1
        assert by_id[EMMA].evidence.anchor_readers == 2
        assert by_id[EMMA].evidence.same_author is False
        assert by_id[RING].evidence.same_author is True
        assert by_id[RING].evidence.shared_series == "Middle-earth"
        assert by_id[DUNE].evidence.co_readers == 1  # user 1 read Hobbit and Dune

    def test_every_suggestion_carries_a_reason(self) -> None:
        engine = DemoEngine(_assets(support=(50, 50, 50, 50, 50)))
        assert all(s.reason for s in engine.similar(HOBBIT, k=4))

    def test_scores_are_descending(self) -> None:
        engine = DemoEngine(_assets(support=(50, 50, 50, 50, 50)))
        scores = [s.evidence.score for s in engine.similar(HOBBIT, k=4)]
        assert scores == sorted(scores, reverse=True)


class TestFind:
    @staticmethod
    def _engine(vector: list[float]) -> DemoEngine:
        return DemoEngine(_assets(), encoder=lambda texts: np.array([vector], dtype=np.float32))

    def test_it_resolves_to_the_nearest_title(self) -> None:
        assert self._engine([1.0, 0.0]).find("hobbit", k=1)[0].isbn == HOBBIT

    def test_the_picker_is_ordered_by_similarity(self) -> None:
        found = self._engine([1.0, 0.0]).find("hobbit", k=3)
        assert [b.isbn for b in found] == [HOBBIT, RING, EMMA]

    def test_an_empty_query_asks_nothing_of_the_encoder(self) -> None:
        def explode(texts: list[str]) -> np.ndarray:
            raise AssertionError("the encoder must not run on an empty query")

        assert DemoEngine(_assets(), encoder=explode).find("   ") == []

    def test_the_resolved_book_carries_its_reader_count(self) -> None:
        assert self._engine([1.0, 0.0]).find("hobbit", k=1)[0].readers == 50

    def test_a_book_the_engine_cannot_answer_for_is_never_offered(self) -> None:
        """Ubik sits below the L34 floor, so `similar` refuses it. Offering it as an anchor
        would be a dead end dressed up as a result — and on the real catalogue this rule is
        what stops *Hoopla — Harry Stein* winning the query "harry potter stein"."""
        engine = self._engine([-1.0, 0.0])  # points straight at Ubik (180 degrees)
        assert engine.similar(UBIK) == []
        assert UBIK not in [b.isbn for b in engine.find("anything", k=5)]

    def test_popularity_only_breaks_a_near_tie(self) -> None:
        """Emma sits 30 degrees off and Fellowship 4, so no readership gap may put Emma
        first: a margin separates near-equal text matches and never reorders across a real
        similarity gap. An additive popularity weight would have failed this."""
        assets = _assets(support=(50, 22, 5_000_000, 50, 50))
        engine = DemoEngine(assets, encoder=lambda texts: np.array([[1.0, 0.0]], dtype=np.float32))
        assert [b.isbn for b in engine.find("hobbit", k=2)] == [HOBBIT, RING]

    def test_popularity_decides_when_the_text_cannot(self) -> None:
        """Two works equidistant from the query: the one more readers mean wins. This is
        the rule that stops 'harry potter stein' resolving to the 21-reader German work."""
        assets = _assets(support=(50, 50, 50, 50, 50))
        # A query exactly between Hobbit (0 deg) and Fellowship (4 deg) -> a genuine tie.
        tie = [float(np.cos(np.deg2rad(2))), float(np.sin(np.deg2rad(2)))]
        assert DemoEngine(assets, encoder=lambda t: np.array([tie], dtype=np.float32)).find("x", k=1)[0].isbn in {
            HOBBIT,
            RING,
        }
        popular = _assets(support=(50, 5000, 50, 50, 50))
        assert DemoEngine(popular, encoder=lambda t: np.array([tie], dtype=np.float32)).find("x", k=1)[0].isbn == RING

    def test_the_floor_does_not_empty_the_picker(self) -> None:
        found = self._engine([-1.0, 0.0]).find("anything", k=5)
        assert [b.isbn for b in found] == [DUNE, EMMA, RING, HOBBIT]


class TestAssetContract:
    """The app must fail with a sentence, not a shape mismatch three frames later."""

    def _write(self, directory, version: int) -> None:
        assets = _assets()
        np.save(directory / "factors.npy", assets.factors)
        np.save(directory / "item_ids.npy", assets.item_ids)
        np.save(directory / "item_support.npy", assets.item_support)
        sp.save_npz(directory / "readers.npz", assets.readers)
        np.save(directory / "lookup_vectors.npy", assets.lookup_vectors)
        np.save(directory / "lookup_ids.npy", assets.lookup_ids)
        np.save(directory / "lookup_support.npy", assets.lookup_support)
        BOOKS.reset_index().to_parquet(directory / "books.parquet", index=False)
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "asset_version": version,
                    "similar_min_support": 20,
                    "encoder_model": "stub",
                    "item_level": "work",
                }
            )
        )

    def test_a_round_trip_reproduces_the_engine(self, tmp_path) -> None:
        self._write(tmp_path, ASSET_VERSION)
        engine = DemoEngine(load_assets(tmp_path))
        assert [s.isbn for s in engine.similar(HOBBIT, k=2)] == [RING, EMMA]
        assert engine.assets.item_level == "work"

    def test_a_stale_version_is_refused_with_the_command_to_fix_it(self, tmp_path) -> None:
        self._write(tmp_path, ASSET_VERSION + 1)
        with pytest.raises(RuntimeError, match="build_app_assets"):
            load_assets(tmp_path)

    def test_missing_assets_do_not_pretend_to_work(self, tmp_path) -> None:
        with pytest.raises((FileNotFoundError, ValueError)):
            load_assets(tmp_path)
