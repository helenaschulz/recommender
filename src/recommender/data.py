"""Data-prep layer: load the raw Book-Crossing CSVs and turn them into an interaction matrix.

This is the only module that knows what the raw files look like. It owns three
responsibilities and nothing else:

1. **Loading and repair.** Three CSVs, one repair step: `Books.csv` contains three
   records whose title holds an unescaped ``\\";`` sequence, which merges title and
   author into a single CSV field and shifts every later column one position left.
   They are repaired, never skipped (see notebook 01, §1).
2. **Flags the rest of the pipeline needs.** ``is_explicit`` (rating > 0 — a 0 in
   Book-Crossing is an interaction without a grade) and ``has_metadata`` (the ISBN
   resolves to a row in ``Books.csv``; 10.3% do not, ledger L14).
3. **Matrix construction.** A user x item CSR matrix over a fixed item universe, so
   that train and test share one index space.

Everything here is fit-free: no statistic computed in this module ever depends on the
holdout. Splitting lives in :mod:`recommender.split`, scoring in :mod:`recommender.models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BOOK_COLUMNS = [
    "Book-Title",
    "Book-Author",
    "Year-Of-Publication",
    "Publisher",
    "Image-URL-S",
    "Image-URL-M",
    "Image-URL-L",
]
#: Size of the published Book-Crossing catalogue. Catalog-Coverage@K is measured
#: against this number so the metric means "share of the catalogue we can reach".
CATALOG_SIZE = 271_360


def find_data_dir(start: Path | None = None) -> Path:
    """Walk up from *start* (default: cwd) until a directory containing ``data/`` is found."""
    root = (start or Path.cwd()).resolve()
    while not (root / "data").is_dir() and root != root.parent:
        root = root.parent
    data = root / "data"
    if not (data / "Ratings.csv").exists():
        raise FileNotFoundError(
            f"Book-Crossing CSVs not found under {data}. Download the Kaggle dataset "
            "'arashnic/book-recommendation-dataset' and place Books.csv, Ratings.csv and "
            "Users.csv there (see README)."
        )
    return data


def project_root(start: Path | None = None) -> Path:
    """The repository root: the first ancestor of *start* (default cwd) holding ``data/``.

    Anything written outside the repo tree (cached vectors, figures) resolves through
    this, so a notebook run from ``notebooks/`` and a script run from the root agree on
    where artefacts live. Without it, the same cache gets rebuilt once per working
    directory.
    """
    root = (start or Path.cwd()).resolve()
    while not (root / "data").is_dir() and root != root.parent:
        root = root.parent
    return root


def repair_shifted_rows(books: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Un-shift the rows whose title+author were merged into one CSV field.

    Returns the repaired frame and the number of rows repaired. Nothing is dropped:
    a silent ``on_bad_lines="skip"`` is exactly the habit this repairs against.
    """
    year = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
    broken = year.isna() & books["Year-Of-Publication"].notna()
    fixed = books.copy()
    for i in books.index[broken]:
        row = books.loc[i]
        title, _, author = str(row["Book-Title"]).partition('\\";')
        fixed.loc[i, BOOK_COLUMNS] = [
            title,
            author.rstrip('"'),
            row["Book-Author"],
            row["Year-Of-Publication"],
            row["Publisher"],
            row["Image-URL-S"],
            row["Image-URL-M"],
        ]
    return fixed, int(broken.sum())


@dataclass(frozen=True)
class BookCrossing:
    """The three raw tables after loading, repair and flagging."""

    ratings: pd.DataFrame  # User-ID, ISBN, Book-Rating, is_explicit, has_metadata
    books: pd.DataFrame  # repaired Books.csv
    users: pd.DataFrame
    n_repaired: int

    @cached_property
    def titles(self) -> pd.Series:
        return self.books.set_index("ISBN")["Book-Title"]

    @cached_property
    def authors(self) -> pd.Series:
        return self.books.set_index("ISBN")["Book-Author"]

    def describe(self, isbn: str) -> str:
        """'Title — Author' for display, or the bare ISBN when metadata is missing."""
        title = self.titles.get(isbn)
        if title is None or (isinstance(title, float) and np.isnan(title)):
            return f"[no metadata: {isbn}]"
        author = self.authors.get(isbn)
        return f"{title} — {author}" if isinstance(author, str) else str(title)


def load(data_dir: Path | None = None) -> BookCrossing:
    """Load, repair and flag the three CSVs."""
    data_dir = data_dir or find_data_dir()
    books_raw = pd.read_csv(data_dir / "Books.csv", dtype=str)
    books, n_repaired = repair_shifted_rows(books_raw)
    ratings = pd.read_csv(
        data_dir / "Ratings.csv",
        dtype={"User-ID": "int64", "ISBN": "str", "Book-Rating": "int64"},
    )
    users = pd.read_csv(
        data_dir / "Users.csv",
        dtype={"User-ID": "int64", "Location": "str", "Age": "float64"},
    )
    ratings = ratings.assign(
        is_explicit=ratings["Book-Rating"] > 0,
        has_metadata=ratings["ISBN"].isin(set(books["ISBN"])),
    )
    return BookCrossing(ratings=ratings, books=books, users=users, n_repaired=n_repaired)


@dataclass(frozen=True)
class Interactions:
    """A user x item CSR matrix plus the index maps that give its rows and columns names.

    ``matrix`` carries interaction *weights*: 1.0 everywhere when binarized (the pinned
    signal decision — an interaction is evidence, graded or not), or the explicit rating
    when built with ``weights="rating"``, which ALS uses for its confidence term.
    """

    matrix: sp.csr_matrix
    user_ids: np.ndarray  # row -> User-ID
    item_ids: np.ndarray  # column -> ISBN

    @cached_property
    def user_index(self) -> dict[int, int]:
        return {u: i for i, u in enumerate(self.user_ids)}

    @cached_property
    def item_index(self) -> dict[str, int]:
        return {b: i for i, b in enumerate(self.item_ids)}

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    @cached_property
    def item_popularity(self) -> np.ndarray:
        """Interactions per item (column counts). The popularity baseline and the
        novelty metric both read this, so it is computed once, on train only."""
        return np.asarray((self.matrix > 0).sum(axis=0)).ravel()


def build_interactions(
    ratings: pd.DataFrame,
    *,
    weights: str = "binary",
    item_ids: np.ndarray | None = None,
    user_ids: np.ndarray | None = None,
) -> Interactions:
    """Build a user x item CSR matrix from a ratings frame.

    Args:
        ratings: rows with ``User-ID``, ``ISBN`` and (for ``weights="rating"``) ``Book-Rating``.
        weights: ``"binary"`` for the pinned interaction signal, ``"rating"`` to carry the
            explicit grade into the cell value.
        item_ids / user_ids: fix the index space explicitly. Pass these whenever two
            matrices must share coordinates (e.g. an ablation built on a subset).

    Duplicate (user, item) pairs are collapsed by taking the maximum, so a user who both
    browsed (0) and graded (8) the same book counts once, with the grade preserved.
    """
    if weights not in {"binary", "rating"}:
        raise ValueError(f"weights must be 'binary' or 'rating', got {weights!r}")

    users = np.asarray(sorted(ratings["User-ID"].unique())) if user_ids is None else np.asarray(user_ids)
    items = np.asarray(sorted(ratings["ISBN"].unique())) if item_ids is None else np.asarray(item_ids)
    u_index = {u: i for i, u in enumerate(users)}
    i_index = {b: i for i, b in enumerate(items)}

    rows = ratings["User-ID"].map(u_index)
    cols = ratings["ISBN"].map(i_index)
    keep = rows.notna() & cols.notna()
    rows = rows[keep].to_numpy(dtype=np.int32)
    cols = cols[keep].to_numpy(dtype=np.int32)
    if weights == "binary":
        vals = np.ones(len(rows), dtype=np.float32)
    else:
        vals = ratings.loc[keep, "Book-Rating"].to_numpy(dtype=np.float32)

    coo = sp.coo_matrix((vals, (rows, cols)), shape=(len(users), len(items)), dtype=np.float32)
    matrix = coo.tocsr()
    matrix.sum_duplicates()
    # Collapse duplicates to a max rather than a sum: repeated rows are the same event.
    if len(rows) != matrix.nnz:
        matrix = sp.csr_matrix(
            (np.minimum(matrix.data, 10.0 if weights == "rating" else 1.0), matrix.indices, matrix.indptr),
            shape=matrix.shape,
        )
    return Interactions(matrix=matrix, user_ids=users, item_ids=items)
