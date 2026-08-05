"""Data-prep tests: the malformed-row repair and the matrix builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data import build_interactions, repair_shifted_rows


def test_repair_unshifts_the_merged_title_and_author() -> None:
    """The real Books.csv damage, reproduced exactly: an unescaped \\"; merges title and
    author into one CSV field, so every later column lands one position to the left."""
    broken = pd.DataFrame(
        {
            "ISBN": ["078946697X"],
            "Book-Title": ['DK Readers: Creating the X-Men\\";Michael Teitelbaum"'],
            "Book-Author": ["2000"],
            "Year-Of-Publication": ["DK Publishing Inc"],
            "Publisher": ["http://images.amazon.com/S.jpg"],
            "Image-URL-S": ["http://images.amazon.com/M.jpg"],
            "Image-URL-M": ["http://images.amazon.com/L.jpg"],
            "Image-URL-L": [None],
        }
    )
    fixed, n = repair_shifted_rows(broken)
    assert n == 1
    row = fixed.iloc[0]
    assert row["Book-Title"] == "DK Readers: Creating the X-Men"
    assert row["Book-Author"] == "Michael Teitelbaum"
    assert row["Year-Of-Publication"] == "2000"
    assert row["Publisher"] == "DK Publishing Inc"
    assert row["Image-URL-S"] == "http://images.amazon.com/S.jpg"


def test_repair_leaves_healthy_rows_alone_and_drops_nothing() -> None:
    healthy = pd.DataFrame(
        {
            "ISBN": ["a", "b"],
            "Book-Title": ["Fine", "Also fine"],
            "Book-Author": ["X", "Y"],
            "Year-Of-Publication": ["2001", "2002"],
            "Publisher": ["P", "P"],
            "Image-URL-S": ["s", "s"],
            "Image-URL-M": ["m", "m"],
            "Image-URL-L": ["l", "l"],
        }
    )
    fixed, n = repair_shifted_rows(healthy)
    assert n == 0
    assert len(fixed) == len(healthy)
    pd.testing.assert_frame_equal(fixed, healthy)


def test_binarized_matrix_keeps_implicit_interactions(toy_ratings: pd.DataFrame) -> None:
    """The pinned signal: a rating of 0 is an interaction and must land in the matrix."""
    inter = build_interactions(toy_ratings, weights="binary")
    user5 = inter.user_index[5]  # user 5 has only implicit rows
    assert inter.matrix[user5].nnz == 3
    assert set(np.unique(inter.matrix.data)) == {1.0}


def test_rating_weights_carry_the_grade(toy_ratings: pd.DataFrame) -> None:
    inter = build_interactions(toy_ratings, weights="rating")
    row, col = inter.user_index[1], inter.item_index["b3"]
    assert inter.matrix[row, col] == 10.0


def test_item_popularity_counts_rows_not_ratings(toy_ratings: pd.DataFrame) -> None:
    inter = build_interactions(toy_ratings, weights="binary")
    # b1 is touched by users 1-5, b7 only by user 4.
    assert inter.item_popularity[inter.item_index["b1"]] == 5
    assert inter.item_popularity[inter.item_index["b7"]] == 1


def test_fixed_item_universe_is_respected(toy_ratings: pd.DataFrame) -> None:
    """Passing item_ids pins the column space so two matrices share coordinates."""
    universe = np.array(["b1", "b2", "zzz"])
    inter = build_interactions(toy_ratings, item_ids=universe)
    assert inter.n_items == 3
    assert inter.item_popularity[inter.item_index["zzz"]] == 0
