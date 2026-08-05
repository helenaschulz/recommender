"""Tiny hand-built fixtures. Offline and deterministic: no CSVs, no network, no downloads.

The fixture is small enough that every expected metric below can be worked out on paper,
which is the point — a metric verified against another implementation of itself proves
nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recommender.data import BookCrossing


@pytest.fixture
def toy_ratings() -> pd.DataFrame:
    """Six users over seven books, mixing implicit (0) and explicit (1-10) rows.

    Designed so eligibility is interesting:
      user 1: 5 explicit, three of them >=8   -> eligible
      user 2: 5 explicit, none >=8            -> not eligible (nothing worth predicting)
      user 3: 4 explicit, one >=8             -> not eligible (profile too thin)
      user 4: 6 explicit, one >=8             -> eligible
      user 5: only implicit interactions      -> not eligible
      user 6: 5 explicit, two >=8             -> eligible
    """
    rows: list[tuple[int, str, int]] = []
    rows += [(1, "b1", 9), (1, "b2", 8), (1, "b3", 10), (1, "b4", 5), (1, "b5", 3), (1, "b6", 0)]
    rows += [(2, "b1", 7), (2, "b2", 6), (2, "b3", 5), (2, "b4", 4), (2, "b5", 7)]
    rows += [(3, "b1", 10), (3, "b2", 4), (3, "b3", 3), (3, "b4", 2)]
    rows += [(4, "b1", 8), (4, "b2", 5), (4, "b3", 6), (4, "b4", 7), (4, "b5", 4), (4, "b7", 3)]
    rows += [(5, "b1", 0), (5, "b2", 0), (5, "b3", 0)]
    rows += [(6, "b2", 9), (6, "b3", 9), (6, "b4", 6), (6, "b5", 5), (6, "b6", 4)]
    return pd.DataFrame(rows, columns=["User-ID", "ISBN", "Book-Rating"])


@pytest.fixture
def toy_books() -> pd.DataFrame:
    """Seven books; b7 deliberately has no counterpart anywhere else."""
    return pd.DataFrame(
        {
            "ISBN": ["b1", "b2", "b3", "b4", "b5", "b6", "b7"],
            "Book-Title": [
                "The Hobbit",
                "The Lord of the Rings",
                "The Silmarillion",
                "Pride and Prejudice",
                "Emma",
                "Der Steppenwolf",
                "Cooking with Fire",
            ],
            "Book-Author": [
                "J.R.R. Tolkien",
                "J.R.R. Tolkien",
                "J.R.R. Tolkien",
                "Jane Austen",
                "Jane Austen",
                "Hermann Hesse",
                "Anonymous",
            ],
            "Year-Of-Publication": ["1937", "1954", "1977", "1813", "1815", "1927", "2001"],
            "Publisher": ["Allen"] * 7,
            "Image-URL-S": ["s"] * 7,
            "Image-URL-M": ["m"] * 7,
            "Image-URL-L": ["l"] * 7,
        }
    )


@pytest.fixture
def toy_catalog(toy_ratings: pd.DataFrame, toy_books: pd.DataFrame) -> BookCrossing:
    ratings = toy_ratings.assign(
        is_explicit=toy_ratings["Book-Rating"] > 0,
        has_metadata=toy_ratings["ISBN"].isin(set(toy_books["ISBN"])),
    )
    users = pd.DataFrame({"User-ID": [1, 2, 3, 4, 5, 6], "Location": ["x"] * 6, "Age": [30.0] * 6})
    return BookCrossing(ratings=ratings, books=toy_books, users=users, n_repaired=0)
