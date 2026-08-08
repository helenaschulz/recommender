"""Tests for the shared evaluation universe.

`build_bench` is the single place that decides what an "item" is, and M12 made that
decision switchable. The properties worth pinning are structural — one coverage
denominator, anchors translated to the current item level, and the ordering that keeps
the canonical per-work text out of the holdout's reach.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recommender.benchmark import build_bench, ceilings, inner_bench
from recommender.data import CATALOG_SIZE, BookCrossing, cluster_works
from recommender.models import ALL_MODELS, WORK_LEVEL_PARAMS, build_model

#: Two editions of one work plus three singletons, so work level actually merges something.
BOOKS = pd.DataFrame(
    {
        "ISBN": ["h1", "h2", "e1", "s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        "Book-Title": [
            "The Hobbit",
            "The Hobbit (Collector's Edition)",
            "Emma",
            "Dune",
            "Solaris",
            "Ubik",
            "Neuromancer",
            "Foundation",
            "Hyperion",
            "Perdido Street Station",
        ],
        "Book-Author": [
            "J. R. R. Tolkien",
            "J.R.R. Tolkien",
            "Jane Austen",
            "Frank Herbert",
            "Stanislaw Lem",
            "Philip K. Dick",
            "William Gibson",
            "Isaac Asimov",
            "Dan Simmons",
            "China Mieville",
        ],
        "Year-Of-Publication": ["1937", "1990", "1815", "1965", "1961", "1969", "1984", "1951", "1989", "2000"],
        "Publisher": ["P"] * 10,
        "Image-URL-S": ["s"] * 10,
        "Image-URL-M": ["m"] * 10,
        "Image-URL-L": ["l"] * 10,
        "series": ["", "Collector's Edition", "", "", "", "", "", "", "", ""],
    }
)

#: Enough graded rows per user that a split can be carved twice: the outer holdout leaves
#: every user still eligible for the inner validation split.
RATED = ["h1", "h2", "e1", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
GRADES = [9, 8, 9, 10, 9, 10, 8, 9, 7, 6]


def _catalog() -> BookCrossing:
    rows: list[tuple[int, str, int]] = []
    for user in range(1, 9):
        rows += [(user, isbn, grade) for isbn, grade in zip(RATED, GRADES, strict=True)]
    ratings = pd.DataFrame(rows, columns=["User-ID", "ISBN", "Book-Rating"])
    ratings = ratings.assign(
        is_explicit=ratings["Book-Rating"] > 0,
        has_metadata=ratings["ISBN"].isin(set(BOOKS["ISBN"])),
    )
    users = pd.DataFrame({"User-ID": list(range(1, 9)), "Location": ["x"] * 8, "Age": [30.0] * 8})
    return BookCrossing(ratings=ratings, books=BOOKS, users=users, n_repaired=0)


def test_isbn_level_bench_keeps_the_isbn_universe() -> None:
    bench = build_bench(work_level=False, catalog=_catalog())
    assert not bench.work_level
    assert bench.item_level == "ISBN"
    assert bench.catalog_size == CATALOG_SIZE
    assert bench.catalog_ids == set(BOOKS["ISBN"])
    assert bench.works is None


def test_work_level_bench_collapses_the_two_editions() -> None:
    catalog = _catalog()
    bench = build_bench(work_level=True, catalog=catalog)
    works = cluster_works(BOOKS)
    assert bench.work_level
    assert bench.catalog_size == works.n_works == 9  # h1 and h2 are one work
    assert bench.catalog_ids == set(works.work_of_isbn)
    assert len(bench.catalog.books) == 9
    # Every item id the split sees is a work id, never an ISBN.
    assert set(bench.split.test["ISBN"]) <= bench.catalog_ids


def test_the_coverage_denominator_is_one_number() -> None:
    """M12 pins a single denominator. If the numerator universe and the denominator ever
    came from different item levels, every coverage cell in the table would be wrong."""
    bench = build_bench(work_level=True, catalog=_catalog())
    assert bench.catalog_size == len(bench.catalog_ids)


def test_anchors_are_translated_to_the_current_item_level() -> None:
    isbn_level = build_bench(work_level=False, catalog=_catalog())
    work_level = build_bench(work_level=True, catalog=_catalog())
    assert all("|" not in a for a in isbn_level.anchors)
    assert all("|" in a or a.startswith("isbn:") for a in work_level.anchors)
    assert len(work_level.anchors) == len(isbn_level.anchors)


def test_canonical_text_is_chosen_without_the_holdout() -> None:
    """The ordering guarantee: at work level the split is drawn before the catalogue, so
    a held-out interaction can never decide which edition represents a work."""
    catalog = _catalog()
    bench = build_bench(work_level=True, catalog=catalog)
    held = set(zip(bench.split.test["User-ID"], bench.split.test["ISBN"], strict=True))
    assert held, "fixture must produce a non-empty holdout for this test to mean anything"

    # Recomputing the support the catalogue used must match train-only support.
    works = bench.works
    train_rows = [
        (user, work) not in held
        for user, work in zip(catalog.ratings["User-ID"], works.of(catalog.ratings["ISBN"]), strict=True)
    ]
    assert sum(train_rows) < len(catalog.ratings), "some rows must actually be excluded"


def test_inner_bench_carves_from_train_and_never_from_test() -> None:
    outer = build_bench(work_level=True, catalog=_catalog())
    inner = inner_bench(outer, seed=43)
    outer_holdout = set(zip(outer.split.test["User-ID"], outer.split.test["ISBN"], strict=True))
    inner_holdout = set(zip(inner.split.test["User-ID"], inner.split.test["ISBN"], strict=True))
    assert inner_holdout.isdisjoint(outer_holdout)
    # The inner train is a subset of the outer train: nothing from the test set re-enters.
    outer_train = set(zip(outer.split.train["User-ID"], outer.split.train["ISBN"], strict=True))
    inner_train = set(zip(inner.split.train["User-ID"], inner.split.train["ISBN"], strict=True))
    assert inner_train <= outer_train


def test_inner_bench_keeps_the_outer_denominator() -> None:
    """A sweep scored against a different denominator than the final table would rank
    configurations on a metric the table does not use."""
    outer = build_bench(work_level=True, catalog=_catalog())
    inner = inner_bench(outer, seed=43)
    assert inner.catalog_size == outer.catalog_size
    assert inner.catalog_ids == outer.catalog_ids
    assert inner.work_level is outer.work_level


@pytest.mark.parametrize("work_level", [False, True])
def test_ceilings_are_measured_on_the_benchs_own_universe(work_level: bool) -> None:
    """L20/L21 recomputed per item level. A ceiling carried over from the other table
    would silently rescale every "share of achievable accuracy" statement."""
    bench = build_bench(work_level=work_level, catalog=_catalog())
    bound = ceilings(bench)
    assert bound.item_level == bench.item_level
    assert bound.n_holdouts == bench.split.n_eligible
    # The union can never be below either component, nor above 1.
    assert bound.union >= max(bound.collaborative, bound.content)
    assert 0.0 <= bound.union <= 1.0


def test_the_collaborative_ceiling_only_counts_items_present_in_train() -> None:
    """Hand-checkable: the fixture's users all rate all ten books, so every held-out item
    is still in some other user's train profile and the ceiling is exactly 1."""
    bench = build_bench(work_level=False, catalog=_catalog())
    assert ceilings(bench).collaborative == pytest.approx(1.0)


def test_work_level_params_are_declared_not_improvised() -> None:
    """M12.2 pins hyperparameters per item level. Anything in the override table must name
    a real model, and asking for a level a model has no override for must still build."""
    assert set(WORK_LEVEL_PARAMS) <= set(ALL_MODELS)
    for name in ("popularity", "item-item", "item-item-explicit", "tfidf"):
        assert build_model(name, work_level=True).name == build_model(name).name


def test_work_level_overrides_reach_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a sweep ever does move a parameter, the override must actually take effect —
    a table that silently ignored it would report tuned numbers from an untuned model."""
    monkeypatch.setitem(WORK_LEVEL_PARAMS, "item-item", {"shrinkage": 123.0})
    assert build_model("item-item", work_level=True).shrinkage == 123.0
    assert build_model("item-item").shrinkage != 123.0


@pytest.mark.parametrize("work_level", [False, True])
def test_bench_is_deterministic(work_level: bool) -> None:
    a = build_bench(work_level=work_level, catalog=_catalog())
    b = build_bench(work_level=work_level, catalog=_catalog())
    pd.testing.assert_frame_equal(a.split.test, b.split.test)
    assert a.catalog_ids == b.catalog_ids
