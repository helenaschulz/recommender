"""The demo app's engine: precomputed assets in, a book and ten grounded reasons out.

Everything the Streamlit layer needs lives here, so ``app/main.py`` stays a thin sheet of
widgets and every rule below is testable offline. Three properties are non-negotiable,
because the app is a **demo, not a product**, and a demo has to be boring to operate in
front of a panel:

1. **No fitting at query time.** ALS takes 90 seconds to fit; nothing that slow may sit
   between a keystroke and a result. ``scripts/build_app_assets.py`` fits once and writes
   the arrays; this module only reads them.
2. **No network at query time.** Vectors, factors and metadata come off disk. The only
   model loaded at runtime is the sentence encoder for the free-text box, and it comes
   from the local Hugging Face cache that the build script warms. Cover images are the
   dataset's 2004 Amazon URLs, which mostly 404 — they load lazily if they load and never
   block anything.
3. **Explanations from structured evidence only, never from a language model.** Every
   sentence this module produces is derived from a count, a string equality or a
   similarity value that is in the assets. An LLM explanation layer is a Part 3 roadmap
   item; an LLM in the demo's hot path would be a latency and a truthfulness risk at the
   same time.

**Why ALS drives it, when ALS loses the comparison table.** The app is an item-to-item
surface, and item-to-item is exactly where ALS wins: with a support floor of 20 its
factors give the best neighbourhoods of any model in the project (ledger L34), while its
HitRate@10 is third of six (L55). That the metric and the product surface disagree is not
an embarrassment to hide behind a better-looking number — it is the finding, and the table
where ALS loses gets shown next to the demo deliberately.

**Fitted on everything, and that is not leakage.** The evaluation models fit on
``split.train`` because a metric measured on data the model has seen is worthless. The app
computes no metric: it is the serving path, where withholding a reader's interactions
would simply make the product worse for no reason. The distinction is stated here and in
the build script rather than left for someone to wonder about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recommender.data import project_root, split_series

#: Where the build script writes and the app reads. Gitignored: the arrays are large and
#: fully regenerable from ``data/``.
ASSETS_SUBDIR = "artifacts/app"

#: Bumped when the on-disk layout changes, so an app started against stale assets fails
#: with a sentence instead of a shape mismatch three frames later.
ASSET_VERSION = 2

#: Candidates fetched per requested slot before filtering. Over-fetching is what lets a
#: full top-10 survive dropping the unnameable and the already-seen.
OVERSAMPLE = 10

#: Free-text **disambiguation** margin, and nothing more than that. Lookup candidates whose
#: cosine is within this much of the best one are treated as a tie and re-ordered by reader
#: count; everything further away keeps its text ranking, untouched.
#:
#: A margin rather than an additive popularity weight, because the additive form has a
#: failure mode worth avoiding: with a large enough readership ratio it can promote a work
#: the query genuinely matches *worse*, and no single weight prevents that. A margin cannot
#: — by construction it never reorders across a real similarity gap.
#:
#: It exists because a reader typing a title means the book *most* readers mean. Without it
#: "harry potter stein" resolves to *Harry Potter und der Stein der Weisen* — 21 readers, a
#: legitimate match for a German query, and a work whose neighbourhood is 21 people's worth
#: of noise — rather than *Harry Potter and the Sorcerer's Stone* with 832, which sits
#: 0.020 lower in cosine.
#:
#: 0.06 is the smallest margin that resolves all four queries L37 and L38 flagged; at 0.10
#: unrelated titles start entering the tie group. Chosen on nine queries, which is a small
#: sample and is stated as such — this is an input affordance for a demo, and it touches
#: **nothing** any published number is measured on. The recommendation ranking never sees it.
LOOKUP_TIE_MARGIN = 0.06


def assets_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / ASSETS_SUBDIR


@dataclass(frozen=True)
class Evidence:
    """What the app knows about *why* a book was suggested. All of it is countable."""

    score: float
    co_readers: int
    anchor_readers: int
    same_author: bool
    shared_series: str = ""


@dataclass(frozen=True)
class Suggestion:
    """One row of the result list, ready for display."""

    isbn: str
    title: str
    author: str
    year: str
    series: str
    image_url: str
    evidence: Evidence
    reason: str


@dataclass(frozen=True)
class Book:
    """A resolved catalogue entry — what the free-text box turns a query into."""

    isbn: str
    title: str
    author: str
    year: str
    series: str
    image_url: str
    readers: int


def reason_sentence(evidence: Evidence, anchor_title: str) -> str:
    """One sentence explaining a suggestion, from structured evidence only.

    The clauses are ordered by what a reader would actually find convincing: shared
    readership first, because "people who read that read this" is the claim the model is
    really making; then the two metadata coincidences; and the raw similarity only when
    there is nothing better to say. The similarity number is never the *whole* sentence
    when a count is available, because "0.57" persuades nobody.

    Deterministic and pure — same evidence, same sentence — which is what makes it
    testable without the app, the assets or the data.
    """
    clauses: list[str] = []
    if evidence.co_readers > 0:
        readers = f"{evidence.co_readers:,} reader" + ("s" if evidence.co_readers != 1 else "")
        clauses.append(f"{readers} of *{anchor_title}* also read this")
    if evidence.same_author:
        clauses.append("same author")
    if evidence.shared_series:
        clauses.append(f"same series ({evidence.shared_series})")

    if not clauses:
        # No shared readers and no metadata overlap: the model is speaking from its own
        # latent geometry and nothing else. Say exactly that rather than dressing it up.
        return f"Close in the model's neighbourhood of *{anchor_title}* (similarity {evidence.score:.2f})."

    sentence = clauses[0][0].upper() + clauses[0][1:]
    if len(clauses) > 1:
        sentence += " — " + ", ".join(clauses[1:])
    return f"{sentence} (similarity {evidence.score:.2f})."


@dataclass(frozen=True)
class DemoAssets:
    """Everything the engine reads, in the shapes it reads them in."""

    factors: np.ndarray  # (n_items, d) L2-normalized ALS item factors
    item_ids: np.ndarray  # (n_items,) ISBN per factor row
    item_support: np.ndarray  # (n_items,) interactions per item, for the L34 floor
    readers: sp.csr_matrix  # (n_items, n_users) — row i lists the readers of item i
    lookup_vectors: np.ndarray  # (n_books, d2) uncentered embeddings, for find_book (L37)
    lookup_ids: np.ndarray  # (n_books,) ISBN per lookup row
    lookup_support: np.ndarray  # (n_books,) interactions per lookup row, for the same floor
    books: pd.DataFrame  # id-indexed metadata: title, author, year, series, image
    similar_min_support: int
    encoder_model: str
    item_level: str = "work"

    @property
    def item_index(self) -> dict[str, int]:
        return {isbn: i for i, isbn in enumerate(self.item_ids.tolist())}


def load_assets(directory: Path | None = None) -> DemoAssets:
    """Read the assets written by ``scripts/build_app_assets.py``.

    Large arrays are memory-mapped: cold start is dominated by page-faulting whatever the
    first query touches rather than by reading ~600 MB up front, which is the difference
    between an app that opens instantly and one a presenter has to apologise for.
    """
    directory = directory or assets_dir()
    meta = pd.read_json(directory / "meta.json", typ="series")
    if int(meta["asset_version"]) != ASSET_VERSION:
        raise RuntimeError(
            f"assets in {directory} are version {meta['asset_version']}, this code expects "
            f"{ASSET_VERSION}. Re-run: python scripts/build_app_assets.py"
        )
    books = pd.read_parquet(directory / "books.parquet").set_index("ISBN")
    return DemoAssets(
        factors=np.load(directory / "factors.npy", mmap_mode="r"),
        item_ids=np.load(directory / "item_ids.npy", allow_pickle=True),
        item_support=np.load(directory / "item_support.npy"),
        readers=sp.load_npz(directory / "readers.npz").tocsr(),
        lookup_vectors=np.load(directory / "lookup_vectors.npy", mmap_mode="r"),
        lookup_ids=np.load(directory / "lookup_ids.npy", allow_pickle=True),
        lookup_support=np.load(directory / "lookup_support.npy"),
        books=books,
        similar_min_support=int(meta["similar_min_support"]),
        encoder_model=str(meta["encoder_model"]),
        item_level=str(meta.get("item_level", "work")),
    )


class DemoEngine:
    """Free text in, ten explained books out. Holds no state beyond the assets."""

    def __init__(self, assets: DemoAssets, *, encoder=None) -> None:
        self.assets = assets
        self._index = assets.item_index
        self._lookup_index = {isbn: i for i, isbn in enumerate(assets.lookup_ids.tolist())}
        self._encoder = encoder

    # -- metadata -----------------------------------------------------------------

    def describe(self, isbn: str) -> Book:
        row = self.assets.books.loc[isbn] if isbn in self.assets.books.index else None
        item = self._index.get(isbn)
        return Book(
            isbn=isbn,
            title=str(row["Book-Title"]) if row is not None else f"[unknown ISBN {isbn}]",
            author=str(row["Book-Author"]) if row is not None else "",
            year=str(row["Year-Of-Publication"]) if row is not None else "",
            series=str(row["series"]) if row is not None else "",
            image_url=str(row["Image-URL-M"]) if row is not None else "",
            readers=int(self.assets.item_support[item]) if item is not None else 0,
        )

    # -- the input path -----------------------------------------------------------

    def find(self, query: str, k: int = 5) -> list[Book]:
        """Free text to catalogue entries, via the *uncentered* embedding vectors.

        Uncentered on purpose (ledger L37): a lookup query is a single point rather than
        an average, so the common direction that centering removes is part of what matches
        it to a title. Centering it pushed "harry potter stein" from rank 2 to rank 5.

        Two rules sit on top of the raw cosine, and they do different jobs.

        **The candidate set is restricted to works the engine can actually answer for** —
        the same support floor L34 pins on ALS similarity. A serving rule, not a tuning
        knob: offering an anchor whose neighbourhood the model would refuse to produce is a
        dead end dressed up as a result. It also removes most of what L38 recorded as
        lookup failure, because those were one- and two-reader books winning the argmax by
        chance — *Hoopla — Harry Stein* beating the actual Harry Potter.

        **Then :data:`LOOKUP_TIE_MARGIN` disambiguates what is left.** Among works whose
        titles match the query almost equally well, prefer the one most readers mean.

        What neither rule fixes is L38's real finding: "herr der ringe" and "hobit tolkien"
        still return nothing relevant, because title+author is three to five words — too
        thin for a multilingual encoder to bridge. That failure is the measured argument
        for the enrichment layer and it survives both rules intact.
        """
        text = query.strip()
        if not text:
            return []
        vector = self._encode(text)
        support = self.assets.lookup_support
        scores = np.asarray(self.assets.lookup_vectors @ vector)
        scores = np.where(support >= self.assets.similar_min_support, scores, -np.inf)
        take = min(k * OVERSAMPLE, scores.size)
        best = np.argpartition(-scores, kth=take - 1)[:take]
        best = best[np.argsort(-scores[best], kind="stable")]
        best = best[np.isfinite(scores[best])]
        if best.size == 0:
            return []

        # The tie-break, applied to the shortlist only: candidates within the margin of the
        # best cosine are re-ordered by readership; the rest keep their text ranking.
        cutoff = scores[best[0]] - LOOKUP_TIE_MARGIN
        tied = [row for row in best.tolist() if scores[row] >= cutoff]
        rest = [row for row in best.tolist() if scores[row] < cutoff]
        tied.sort(key=lambda row: (-support[row], -scores[row]))

        seen: set[str] = set()
        found: list[Book] = []
        for row in tied + rest:
            isbn = str(self.assets.lookup_ids[row])
            if isbn in seen:
                continue
            seen.add(isbn)
            found.append(self.describe(isbn))
            if len(found) == k:
                break
        return found

    def _encode(self, text: str) -> np.ndarray:
        if self._encoder is None:
            from recommender.models.embeddings import sentence_transformer_encoder

            self._encoder = sentence_transformer_encoder(self.assets.encoder_model)
        vector = np.asarray(self._encoder([text.lower()]), dtype=np.float32).ravel()
        return vector / (np.linalg.norm(vector) or 1.0)

    # -- the answer ---------------------------------------------------------------

    def similar(self, isbn: str, k: int = 10) -> list[Suggestion]:
        """The k most similar books to *isbn*, deduplicated by work, each with a reason.

        Empty when the anchor has no ALS factor, **or when the anchor itself sits below the
        support floor** — an honest "we do not know this book well enough" beats a
        fabricated neighbourhood.

        The second condition is where this deliberately goes further than
        :meth:`recommender.models.als.ALSRecommender.similar_items`, which applies the L34
        floor to candidates only. That is right for the gallery, whose anchors are chosen
        to be well-supported, and wrong for an app where a visitor can type anything: the
        argument in L34 — that a factor built from one interaction is a noise direction —
        applies just as much to the vector being queried as to the vectors being ranked. It
        is also what makes the rule in :meth:`find` coherent: never offer an anchor, then
        actually refuse it.
        """
        anchor = self._index.get(isbn)
        if anchor is None or self.assets.item_support[anchor] < self.assets.similar_min_support:
            return []

        scores = np.asarray(self.assets.factors @ np.asarray(self.assets.factors[anchor]))
        # The L34 support floor: 196k single-interaction items have noise-direction
        # factors, and the best of 196k coincidences reaches cosine 0.95 in 128 dimensions.
        scores[self.assets.item_support < self.assets.similar_min_support] = -np.inf
        scores[anchor] = -np.inf

        take = min(k * OVERSAMPLE, scores.size - 1)
        best = np.argpartition(-scores, kth=take - 1)[:take]
        best = best[np.argsort(-scores[best], kind="stable")]

        # The reason sentence names the anchor, and the canonical title keeps its edition
        # parenthetical -- "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"
        # reads badly ten times in a row. Display drops it; the work id is unchanged.
        anchor_title = split_series(self.describe(isbn).title)[0]
        anchor_readers = self._readers_of(anchor)
        anchor_book = self.assets.books.loc[isbn] if isbn in self.assets.books.index else None

        seen = {isbn}
        out: list[Suggestion] = []
        for row in best.tolist():
            if not np.isfinite(scores[row]):
                continue
            other = str(self.assets.item_ids[row])
            if other in seen or other not in self.assets.books.index:
                # A book we cannot name is a book we cannot show (ledger L46): 10.3% of
                # interactions point at ISBNs with no catalogue row, and a card reading
                # "[unknown 0432534220]" is worse than one fewer suggestion.
                continue
            seen.add(other)
            book = self.describe(other)
            evidence = Evidence(
                score=float(scores[row]),
                co_readers=int(np.intersect1d(anchor_readers, self._readers_of(row), assume_unique=True).size),
                anchor_readers=int(anchor_readers.size),
                same_author=bool(
                    anchor_book is not None and book.author and book.author == str(anchor_book["Book-Author"])
                ),
                shared_series=book.series
                if anchor_book is not None and book.series and book.series == str(anchor_book["series"])
                else "",
            )
            out.append(
                Suggestion(
                    isbn=book.isbn,
                    title=book.title,
                    author=book.author,
                    year=book.year,
                    series=book.series,
                    image_url=book.image_url,
                    evidence=evidence,
                    reason=reason_sentence(evidence, anchor_title),
                )
            )
            if len(out) == k:
                break
        return out

    def _readers_of(self, item_row: int) -> np.ndarray:
        readers = self.assets.readers
        lo, hi = readers.indptr[item_row], readers.indptr[item_row + 1]
        return readers.indices[lo:hi]
