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
3. **Edition clustering.** ``work_id`` groups the ISBNs that are the same *work*, and
   ``series`` holds the edition/series packaging parsed out of the title. See
   :func:`cluster_works` — this is where the ledger's L31/L39 duplicate problem is
   addressed, upstream of every model.
4. **Matrix construction.** A user x item CSR matrix over a fixed item universe, so
   that train and test share one index space.

Everything here is fit-free: no statistic computed in this module ever depends on the
holdout. Splitting lives in :mod:`recommender.split`, scoring in :mod:`recommender.models`.
"""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
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


# --------------------------------------------------------------------------------------
# Edition clustering: which ISBNs are the same work
# --------------------------------------------------------------------------------------
#
# The problem, measured: *Crime and Punishment* by Dostoevsky is 21 ISBNs sharing 141
# interactions, the strongest edition holding 40 and 13 of them sitting below the min-5
# threshold standard preprocessing applies. Fragmentation dilutes every co-occurrence
# count upstream of the models, and it is why a third of the TF-IDF recommendation slots
# are another edition of a book the user already holds (L31, L39). Clustering first is
# worth 18% of item-item's hit rate (L44).
#
# **The key** (pinned by Helena + Clody, 04.08.2026): the title with its trailing
# parenthetical stripped and normalized (HTML-unescaped, lower-cased, whitespace
# collapsed), plus the author's last-name token, lower-cased. Title alone would merge
# *Crime and Punishment* by Dostoevsky with *Crime and Punishment* by Ali Brownlie — a
# school textbook about crime, a genuinely different work. The author's *last name* alone
# (rather than the full string) is what absorbs "Fyodor" / "Fedor" / "Feodor" /
# "Fyodor M.".
#
# **One measured extension to that key.** The pinned key does not do what the pinned test
# asks of it: "Dostoevsky" and "Dostoyevsky" differ in the last-name token itself, so four
# of the five spellings merge and the fifth does not. Transliteration variants are
# resolved by a second pass that merges last-name tokens *within an identical normalized
# title* when they are within one edit of each other and at least
# ``MIN_VARIANT_NAME_LENGTH`` characters long. It is deliberately narrow: it can never
# join two different titles, and the length floor keeps short near-collisions (keats /
# yeats) apart. See ledger L41 for what it costs and `STRATEGY.md` for the deviation note.

#: Below this many characters, a one-edit difference between two surnames is as likely to
#: be two people as one spelling. Dostoevsky/Dostoyevsky is 10-11 characters.
MIN_VARIANT_NAME_LENGTH = 6

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def split_series(title: str) -> tuple[str, str]:
    """Split a raw title into ``(title without trailing parentheticals, series text)``.

    The parenthetical is edition packaging — "(Penguin Classics)", "(Book 1)", "(Signet
    Classics (Paperback))" — not part of the work's name. It is parsed out rather than
    thrown away: 74,233 catalogue rows carry one, and it is display metadata (and a
    candidate content feature) in its own right.

    Nesting is handled by matching brackets from the right, so "(Signet Classics
    (Paperback))" comes off as one unit rather than leaving a dangling "(Signet Classics".
    """
    if not isinstance(title, str):
        return "", ""
    text = html.unescape(title).rstrip()
    found: list[str] = []
    while text.endswith(")"):
        depth = 0
        cut = None
        for i in range(len(text) - 1, -1, -1):
            if text[i] == ")":
                depth += 1
            elif text[i] == "(":
                depth -= 1
                if depth == 0:
                    cut = i
                    break
        # cut == 0 means the whole title is parenthesised; that is the title, not a series.
        if not cut:
            break
        found.append(text[cut + 1 : -1].strip())
        text = text[:cut].rstrip()
    return text, " | ".join(reversed(found))


def normalize_title(title: str) -> str:
    """The clustering form of a title: series stripped, unescaped, lower-cased, collapsed."""
    base, _ = split_series(title)
    return _WHITESPACE.sub(" ", base).strip().lower()


def author_last_name(author: str) -> str:
    """The last alphanumeric token of an author string, lower-cased.

    Taking the *last* token beats splitting on a comma: the catalogue's comma rows are
    mostly misplaced suffixes ("Mark E., Jr. Neely", "Rush H., III Limbaugh"), where the
    surname still sits at the end.
    """
    if not isinstance(author, str):
        return ""
    tokens = [_NON_WORD.sub("", token) for token in html.unescape(author).strip().lower().split()]
    tokens = [token for token in tokens if token]
    return tokens[-1] if tokens else ""


def _within_one_edit(a: str, b: str) -> bool:
    """Levenshtein(a, b) <= 1, without building the DP table."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long[i]:
        i += 1
    return short[i:] == long[i + 1 :]


def _merge_name_variants(titles: pd.Series, names: pd.Series) -> dict[tuple[str, str], str]:
    """Map ``(title, last name)`` to the canonical last name of its spelling group.

    Union-find over the surnames that share a normalized title. The canonical member is
    the spelling carried by the most editions, with an alphabetical tie-break — so the
    mapping is deterministic and independent of row order, and the surviving work id
    reads ``dostoevsky`` rather than the ``dostoevdky`` that would win alphabetically.
    """
    by_title: dict[str, Counter[str]] = defaultdict(Counter)
    for title, name in zip(titles, names, strict=True):
        if title and name:
            by_title[title][name] += 1

    canonical: dict[tuple[str, str], str] = {}
    for title, name_counts in by_title.items():
        if len(name_counts) < 2:
            continue
        group = sorted(name_counts)
        parent = {name: name for name in group}

        def find(x: str, parent: dict[str, str] = parent) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, left in enumerate(group):
            if len(left) < MIN_VARIANT_NAME_LENGTH:
                continue
            for right in group[i + 1 :]:
                if len(right) >= MIN_VARIANT_NAME_LENGTH and _within_one_edit(left, right):
                    parent[find(left)] = find(right)

        components: dict[str, list[str]] = defaultdict(list)
        for name in group:
            components[find(name)].append(name)
        for members in components.values():
            if len(members) > 1:
                winner = min(members, key=lambda name: (-name_counts[name], name))
                for name in members:
                    canonical[(title, name)] = winner
    return canonical


@dataclass(frozen=True)
class Works:
    """The ISBN -> work mapping, plus the series text parsed out of each title.

    A *work* is what a reader means by "the book"; an ISBN is one edition of it. Anything
    the mapping cannot resolve — an ISBN outside ``Books.csv``, an empty title, a missing
    author — becomes its own single-ISBN work rather than being lumped together, so an
    unknown never silently merges with another unknown.
    """

    work_of_isbn: pd.Series  # index ISBN -> work_id
    series_of_isbn: pd.Series  # index ISBN -> series text ("" when the title has none)
    label_of_work: pd.Series  # work_id -> a representative "Title — Author" for display

    @property
    def n_works(self) -> int:
        return int(self.work_of_isbn.nunique())

    @cached_property
    def _lookup(self) -> dict[str, str]:
        """Plain dict rather than the Series. The serving path calls :meth:`of` once per
        user with a handful of ISBNs, where building a pandas object per call costs more
        than the lookup itself."""
        return self.work_of_isbn.to_dict()

    def of(self, isbns: np.ndarray | pd.Series | list[str]) -> np.ndarray:
        """Work ids for arbitrary ISBNs; unknown ISBNs become their own work."""
        lookup = self._lookup
        values = isbns.to_numpy() if isinstance(isbns, pd.Series) else isbns
        return np.array([lookup.get(isbn, f"isbn:{isbn}") for isbn in values], dtype=object)

    def describe(self, work_id: str) -> str:
        label = self.label_of_work.get(work_id)
        return str(label) if isinstance(label, str) else f"[no metadata: {work_id}]"


def cluster_works(books: pd.DataFrame, *, merge_author_variants: bool = True) -> Works:
    """Cluster the catalogue's ISBNs into works. Pure function of ``Books.csv``.

    Args:
        merge_author_variants: run the transliteration pass described above. Turning it
            off yields exactly the pinned title+last-name key, which is the comparison
            the ledger reports the extension against.
    """
    isbns = books["ISBN"].to_numpy(dtype=object)
    split = [split_series(title) for title in books["Book-Title"]]
    titles = pd.Series([_WHITESPACE.sub(" ", base).strip().lower() for base, _ in split], index=books.index)
    series = pd.Series([parens for _, parens in split], index=books.index)
    names = books["Book-Author"].map(author_last_name)

    if merge_author_variants:
        canonical = _merge_name_variants(titles, names)
        if canonical:
            names = pd.Series(
                [canonical.get((t, n), n) for t, n in zip(titles, names, strict=True)],
                index=books.index,
            )

    keyed = (titles != "") & (names != "")
    work_ids = np.where(keyed, titles + "|" + names, pd.Series("isbn:", index=books.index) + books["ISBN"])

    work_of_isbn = pd.Series(work_ids, index=pd.Index(isbns, name="ISBN"), dtype=object)
    series_of_isbn = pd.Series(series.to_numpy(), index=pd.Index(isbns, name="ISBN"), dtype=object)
    labels = (
        books.assign(_work=work_ids, _label=books["Book-Title"].astype(str) + " — " + books["Book-Author"].astype(str))
        .drop_duplicates("_work")
        .set_index("_work")["_label"]
    )
    return Works(work_of_isbn=work_of_isbn, series_of_isbn=series_of_isbn, label_of_work=labels)


def to_work_level(ratings: pd.DataFrame, works: Works) -> pd.DataFrame:
    """Re-key a ratings frame from ISBNs to work ids.

    The work id lands in the ``ISBN`` column on purpose: every downstream layer (split,
    matrix builder, metrics) treats that column as an opaque item id, so the work-level
    experiment reuses the *identical* code path rather than a parallel implementation of
    it. A user who rated two editions of the same work collapses to one row carrying the
    **maximum** rating — the same rule ``build_interactions`` applies to duplicate cells.
    """
    rekeyed = ratings.assign(ISBN=works.of(ratings["ISBN"]))
    collapsed = rekeyed.groupby(["User-ID", "ISBN"], as_index=False, sort=False)["Book-Rating"].max()
    return collapsed.assign(
        is_explicit=collapsed["Book-Rating"] > 0,
        has_metadata=~collapsed["ISBN"].str.startswith("isbn:"),
    )


@dataclass(frozen=True)
class BookCrossing:
    """The three raw tables after loading, repair and flagging."""

    ratings: pd.DataFrame  # User-ID, ISBN, Book-Rating, is_explicit, has_metadata
    books: pd.DataFrame  # repaired Books.csv, plus the parsed `series` column
    users: pd.DataFrame
    n_repaired: int

    @cached_property
    def works(self) -> Works:
        """The edition clustering. Computed once, on first use (~3s over 271k rows)."""
        return cluster_works(self.books)

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
    # The trailing parenthetical is edition/series packaging. It leaves the title text
    # (which is what clusters editions) and becomes a field of its own.
    books = books.assign(series=[split_series(title)[1] for title in books["Book-Title"]])
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
