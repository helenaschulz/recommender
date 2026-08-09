"""The demo app's engine: precomputed assets in, a book and ten grounded reasons out.

Everything the Streamlit layer needs lives here, so ``app/main.py`` stays a thin sheet of
widgets and every rule below is testable offline. Three properties are non-negotiable,
because the app is a **demo, not a product**, and a demo has to be boring to operate in
front of a reviewer:

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

import html
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
ASSET_VERSION = 3

#: Tail cutoff, **off by default: it was reverted on 08.08.2026 after seeing it run.**
#: Ten results are what the demo shows; a list that ends at four reads as a broken app in
#: front of a reviewer, whatever the ledger says about the tail. Set it to 0.55 (or pass
#: ``tau=`` per call) to get the behaviour M14.5 measured. The measurement stands as ledger
#: **L68** and is worth saying out loud — "ten slots are a layout choice, not a claim that
#: ten good neighbours exist" — it simply is not wired into the product.
#:
#: The rest of this docstring records what was measured, because the finding survives the
#: reversal and the alternatives should not have to be re-derived.
#:
#: Ten slots were a UI choice, never a claim that ten good neighbours exist. *The Hobbit*
#: ran out after four and finished with *Magic the Gathering: Arena*; *The Da Vinci Code*
#: finished with three books sharing a tenth of its top match's score.
#:
#: **Relative, and that is forced rather than chosen.** L63 measures that absolute cosines
#: are not comparable across anchors — 0.49 is a good score for a 700-reader anchor and a
#: meaningless one for a 25-reader anchor — so an absolute cutoff would gut a
#: well-supported list and leave a thin one whole. A fraction of the anchor's *own* top
#: score has no such problem.
#:
#: Two alternatives were measured and rejected, both in `scripts/analyze_truncation.py`:
#: the largest consecutive gap (an elbow) cuts 76% of all slots and leaves a median list of
#: two, and a threshold on co-readers-per-anchor-reader cannot be a truncation at all,
#: because the score does not order the evidence — mean Spearman 0.39, and in 18.3% of
#: lists some slot carries at least twice the evidence of everything above it. Cutting at
#: the first weak slot would have thrown away *The Secret Life of Bees* at rank 9 under
#: *The Lovely Bones*, which 185 of its 1,295 readers also read.
#:
#: 0.55 removes 1.4% of slots over 300 random anchors: a targeted fix for anchors with a
#: strong head, not a broad change. It does **not** rescue *Guns, Germs, and Steel* — every
#: one of its slots is within 76% of its top score, and all of them are thin. That anchor is
#: the support floor's problem, not truncation's, and conflating the two would be wrong.
#:
#: Where it was aggressive it was *too* aggressive for a demo: *Harry Potter* ended after
#: the four sequels and *Bridget Jones's Diary* after four. Defensible as a claim about
#: evidence, wrong as a product surface — which is exactly the sort of thing that only
#: shows up once it is on screen.
SCORE_TRUNCATION_TAU = 0.0

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

#: How far below the best match a lookup candidate may sit and still be **offered** (M17.4).
#: A different question from :data:`LOOKUP_TIE_MARGIN`, and the two must not be confused:
#: the tie margin decides *which book the query resolves to*, this one decides *which
#: alternatives are worth showing beside it*. Everything past it is dropped from the
#: shortlist, so the picker is sometimes one row and sometimes five.
#:
#: The problem it solves is a credibility one, not a metric one. ``find`` returned five
#: candidates whatever their score, so "little prince" offered *A Little Princess*, a John
#: Saul and a Stephen King next to the right answer; a visibly absurd option makes the good
#: one beside it look like a coincidence.
#:
#: **Derived, in `scripts/analyze_picker_margin.py`, rather than picked.** 300 answerable
#: works are queried by their own titles, a returned candidate counts as on-target when its
#: title contains the query's or vice versa, and rank 1 is excluded because it is on-target
#: by construction. On-target alternatives sit a median 0.032 below the best match,
#: off-target ones 0.212. The separation between the two classes peaks on a **plateau**, not
#: a spike — 0.08 to 0.13 all buy within a point of each other — so the argmax was re-run on
#: six independent samples: median 0.118, range 0.102 to 0.153. 0.12 is that median at the
#: resolution the sample supports, and the third decimal is deliberately not claimed.
#:
#: **It cannot interact with the tie margin, by construction.** At twice 0.06 it never cuts
#: into the tie group, so the group the readership rule re-orders is always whole and **the
#: book a query resolves to cannot change** — only the alternatives listed under it. That is
#: what keeps this a display-side change to a serving path, and it is asserted in the tests
#: rather than left as a claim.
#:
#: What it does not fix: `"Guns Germs Steel"` keeps all five candidates, because all five are
#: Danielle Steel novels within 0.06 of each other. That is L38's under-determined-query
#: finding and no cutoff relative to the best match can see it — the best match is already
#: wrong.
PICKER_MARGIN = 0.12


def assets_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / ASSETS_SUBDIR


def repair_encoding(text: str) -> str:
    """Undo one round of UTF-8-read-as-latin-1, or return the string untouched.

    The 2004 crawl stored some rows already double-encoded, so ``Books.csv`` literally
    contains ``Antoine de Saint-ExupÃ©ry`` — the bytes of *é* rendered as two latin-1
    characters. **The corruption is in the source file, not in how this project reads it**;
    ``pd.read_csv`` is decoding correct UTF-8 whose content happens to be mojibake.

    Rather than a table of substitutions, the inverse of the original mistake: encode back to
    latin-1, decode as UTF-8. **Its own failure is the guard.** A string that was never
    double-encoded either has no latin-1 encoding (rare here) or produces bytes that are not
    valid UTF-8 — a genuine *é* is the single byte ``0xE9``, which is an invalid UTF-8
    sequence on its own — so it raises and is returned unchanged. Nothing is guessed and
    nothing is repaired that does not round-trip exactly.

    Measured over all 469,252 title and author strings in the shipped catalogue (M17
    follow-up): **3,234 change, and every one of them carries the ``Ã``/``Â`` signature** —
    zero strings change without it, which is the false-positive check. One pass is also a
    fixed point: no string needs a second, and no repaired string repairs again, so this is
    idempotent and needs no loop.

    **Display only, and that is load-bearing rather than a disclaimer.** The work key is
    built from the *corrupted* string (``the little prince|saintexupãry``), so repairing at
    load time would move the edition merge and with it published numbers. Repairing here
    moves nothing: no id, key, count or score is derived from what :meth:`DemoEngine.describe`
    returns. The one thing downstream that reads this text is the ``same_author`` comparison,
    and it gets *both* sides from ``describe`` — see there.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def same_author(left: str, right: str) -> bool:
    """Do these two author strings name the same person, ignoring case and padding?

    The catalogue holds ``ANNE RICE`` next to ``Anne Rice``, ``CHUCK PALAHNIUK`` next to
    ``Chuck Palahniuk`` and ``J.R.R. TOLKIEN`` next to ``J.R.R. Tolkien``, because a work's
    displayed author comes from its most-interacted *edition* and different editions were
    catalogued by different people. An exact string comparison therefore dropped the "same
    author" tag from exactly the rows where it was most obviously true — *The Vampire
    Lestat* at rank 1 under *Interview with the Vampire* showed none while rank 2 did.

    Casefold rather than lower, because it is the comparison that also folds ``ß`` and the
    Turkish dotted I, and this catalogue is multilingual. Deliberately **not** a fuzzy
    match: this decides a displayed tag, and "probably the same author" is not a claim the
    app should make from a string edit distance. The work key already carries the one
    fuzzy rule this project accepted, and it is measured (ledger L41/L42).

    This changes the *explanation* only. No ranking, and therefore no published number,
    depends on it.
    """
    return bool(left) and left.strip().casefold() == right.strip().casefold()


@dataclass(frozen=True)
class Evidence:
    """What the app knows about *why* a book was suggested. All of it is countable.

    :attr:`shared_series` is **computed and not displayed** since M17.6. It is kept because
    it is a fact about the row and the field it is derived from is real; what is not real is
    the name. ``series`` holds the title's trailing parenthetical, which is a series 27% of
    the time and a publisher imprint, a format or an edition number the rest — *Penguin
    Classics* covers 378 works, *Dover Thrift Editions* 268, and *Dune* carries *Remembering
    Tomorrow*. A tag reading "same series (Penguin Classics)" is a false claim printed as
    fact, and the comparison also under-fires on the real thing, because one series appears
    as ``Vampire Chronicles (Paperback)``, ``The Vampire Chronicles, Book 6`` and ``Vampire
    Chronicles, No 5``. It comes back when a series is an entity rather than a substring;
    that is on the roadmap and is a data-layer project, not a surface fix.
    """

    score: float
    co_readers: int
    anchor_readers: int
    same_author: bool
    shared_series: str = ""


@dataclass(frozen=True)
class Suggestion:
    """One row of the result list, ready for display.

    ``image_url`` was dropped in M17.7: the covers are the dataset's 2004 Amazon URLs, M15
    cut them from the screen, and nothing has rendered the field since. The build script
    still writes the column — removing it there would change the asset layout, and forcing a
    rebuild days before a demo costs more than the megabyte it saves, especially since a
    rebuild is precisely what moved the lookup between two screenshots in M17.9.
    """

    isbn: str
    title: str
    author: str
    year: str
    series: str
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
    readers: int


def reason_sentence(evidence: Evidence, anchor_title: str) -> str:
    """One sentence explaining a suggestion, from structured evidence only.

    **This renders the anchor report, not the screen** (M17.7, the decision, so nobody has to
    re-derive it). ``scripts/run_demo_anchors.py --reasons`` is its only caller; the app
    builds its rows in ``app/main.py`` from the same :class:`Evidence`, structurally — bar,
    score, counts, tags — and has never read this string. Two renderers of one row is a
    liability, and the choice was to name which is which rather than to collapse them,
    because they answer to different constraints and one of them would have to lie:

    - **the screen shows one list at a time**, so the similarity there is the sort key and
      is displayed (M17.1);
    - **the report puts eleven anchors' lists in one document**, which is exactly the
      cross-anchor comparison L63 says a cosine cannot support. The sentence therefore stays
      a count-only claim, and the report carries the number in its own ``sim`` column, where
      it reads as a column rather than as a sentence's conclusion.

    So the divergence is deliberate, and this docstring is the place it is written down. If
    the app ever renders prose, this is the function to reuse — and the similarity clause is
    the thing to revisit first.

    The clauses are ordered by what a reader would actually find convincing: shared
    readership first, because "people who read that read this" is the claim the model is
    really making; then the metadata coincidence.

    **The raw similarity is printed only when there is nothing else to say** (M14.6). It
    used to be appended to every sentence, and the 14.1 sweep is the argument for removing
    it: a cosine of 0.49 against a 25-reader anchor and 0.49 against a 700-reader anchor
    are the same number meaning different things, so putting it in a *report* invites
    precisely the comparison it cannot support. A count does not have that problem: "169
    readers of X also read this" means the same thing on every anchor.

    Slightly wider than M14.6 as written, which asked for the number to go "where a count
    is available": it goes wherever *any* clause exists, including the metadata-only case
    ("Same author"). The value stays in :class:`Evidence`, so nothing that measures anything
    loses access to it.

    The "same series" clause went with the tag in M17.6 — see :class:`Evidence` for why the
    field does not mean what it is called.

    Deterministic and pure — same evidence, same sentence — which is what makes it
    testable without the app, the assets or the data.
    """
    clauses: list[str] = []
    if evidence.co_readers > 0:
        readers = f"{evidence.co_readers:,} reader" + ("s" if evidence.co_readers != 1 else "")
        clauses.append(f"{readers} of *{anchor_title}* also read this")
    if evidence.same_author:
        clauses.append("same author")

    if not clauses:
        # No shared readers and no metadata overlap: the model is speaking from its own
        # latent geometry and nothing else. Say exactly that rather than dressing it up.
        return f"Close in the model's neighbourhood of *{anchor_title}* (similarity {evidence.score:.2f})."

    sentence = clauses[0][0].upper() + clauses[0][1:]
    if len(clauses) > 1:
        sentence += " — " + ", ".join(clauses[1:])
    return f"{sentence}."


def _truncate(suggestions: list[Suggestion], tau: float) -> list[Suggestion]:
    """Drop the tail once the score falls below *tau* times the best one.

    A prefix, not a filter: the first slot under the bar ends the list, and nothing behind
    it is promoted. Truncating rather than filtering keeps the model's ranking intact — a
    filter that reached past a weak slot to keep a strong one would be a re-ranking, and a
    re-ranking is a different product decision from "stop when the evidence runs out".
    """
    if tau <= 0 or not suggestions:
        return suggestions
    cutoff = suggestions[0].evidence.score * tau
    kept = 1
    while kept < len(suggestions) and suggestions[kept].evidence.score >= cutoff:
        kept += 1
    return suggestions[:kept]


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
    similar_min_support: int  # the L34 floor, applied to *candidates*
    encoder_model: str
    item_level: str = "work"
    #: The floor applied to the **anchor**, which is a different question from the floor
    #: applied to the candidates and was one number until M14.2 measured them apart.
    #: ``None`` means "the same as :attr:`similar_min_support`", i.e. the M13 behaviour.
    #:
    #: Why they must be separable: L34 pins the candidate floor at 20 and it is validated —
    #: raising it removes genuinely relevant but thinly-read books, and the eleven-anchor
    #: read shows exactly that (*Dune* loses *Heretics of Dune*, *Harry Potter* loses
    #: *Quidditch Through the Ages*). L63 is a claim about the **anchor**: a factor fitted
    #: from 25 interactions cannot support any similarity, however well-read the candidate
    #: is. Raising the anchor floor removes *anchors*; it never degrades a surviving
    #: anchor's list, because it does not touch the candidate pool.
    anchor_min_support: int | None = None

    @property
    def anchor_floor(self) -> int:
        return self.similar_min_support if self.anchor_min_support is None else self.anchor_min_support

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
        anchor_min_support=int(meta["anchor_min_support"]) if "anchor_min_support" in meta else None,
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
        """Catalogue metadata for one work, with the dataset's two text defects undone.

        **HTML entities.** The 2004 crawl stored titles as they appeared in HTML, so
        ``Books.csv`` literally contains ``Angels &amp; Demons`` and ``Marley &amp; Me``. The
        *work key* has always unescaped them (``split_series`` does it), which is why the id
        reads ``angels & demons|brown`` — but the displayed title did not, so the app printed
        the entity on screen from M13 until M14.3.

        **Double encoding**, the same defect one layer down and fixed the same way: the file
        also holds ``Antoine de Saint-ExupÃ©ry``. See :func:`repair_encoding` for why the
        repair cannot damage a string that was never broken. It is rare and it is on screen:
        **1 of the 2,508 works that can be an anchor** — *The Little Prince* — and **12 of
        the 7,541 that can appear as a candidate**, among them *Le Petit Prince*, two Spanish
        *Harry Potter* volumes and *Smiley's people* by ``John Le CarrÃ©``.

        Doing both here rather than in the app means every consumer gets the clean string:
        the screen, the anchor report, and the anchor name inside a reason sentence.

        **Display only. No id, key, count or score is derived from this text** — the work key
        is built upstream from the raw column and still carries both defects, deliberately.
        The one comparison that reads these strings is ``same_author`` in :meth:`similar`,
        and it takes *both* sides from this method, so the two are always cleaned to the same
        standard. That is the M14.3 lesson: a comparison between a cleaned string and a raw
        one is a bug waiting for the right pair of editions.
        """
        row = self.assets.books.loc[isbn] if isbn in self.assets.books.index else None
        if row is None:
            return Book(
                isbn=isbn, title=f"[unknown ISBN {isbn}]", author="", year="", series="", readers=0
            )
        item = self._index.get(isbn)

        def clean(value: object) -> str:
            return repair_encoding(html.unescape(str(value)))

        return Book(
            isbn=isbn,
            title=clean(row["Book-Title"]),
            author=clean(row["Book-Author"]),
            year=str(row["Year-Of-Publication"]),
            series=clean(row["series"]),
            readers=int(self.assets.item_support[item]) if item is not None else 0,
        )

    # -- the input path -----------------------------------------------------------

    def find(self, query: str, k: int = 5, *, margin: float | None = None) -> list[Book]:
        """Free text to catalogue entries, via the *uncentered* embedding vectors.

        Uncentered on purpose (ledger L37): a lookup query is a single point rather than
        an average, so the common direction that centering removes is part of what matches
        it to a title. Centering it pushed "harry potter stein" from rank 2 to rank 5.

        Three rules sit on top of the raw cosine, and they do different jobs.

        **The candidate set is restricted to works the engine can actually answer for** —
        the same support floor L34 pins on ALS similarity. A serving rule, not a tuning
        knob: offering an anchor whose neighbourhood the model would refuse to produce is a
        dead end dressed up as a result. It also removes most of what L38 recorded as
        lookup failure, because those were one- and two-reader books winning the argmax by
        chance — *Hoopla — Harry Stein* beating the actual Harry Potter.

        **Then :data:`PICKER_MARGIN` drops what is not worth offering.** Returning five
        candidates whatever their score put *A Little Princess* and a Stephen King next to
        *The Little Prince*, which costs the right answer its credibility. Fewer than *k*
        results is therefore normal and not a failure — often exactly one.

        **Then :data:`LOOKUP_TIE_MARGIN` disambiguates what is left.** Among works whose
        titles match the query almost equally well, prefer the one most readers mean. It
        runs *inside* the picker margin — 0.06 against 0.12 — so the tie group is never cut
        into and the book a query resolves to is exactly what it was before M17.4.

        What none of the three fixes is L38's real finding: "herr der ringe" and "hobit
        tolkien" still return nothing relevant, because title+author is three to five words
        — too thin for a multilingual encoder to bridge. That failure is the measured
        argument for the enrichment layer and it survives all three rules intact.
        """
        text = query.strip()
        if not text:
            return []
        vector = self._encode(text)
        support = self.assets.lookup_support
        scores = np.asarray(self.assets.lookup_vectors @ vector)
        # The *anchor* floor, because what this method offers is an anchor. Keeping the two
        # in step is the rule that makes `find` coherent: never offer a book, then refuse it.
        scores = np.where(support >= self.assets.anchor_floor, scores, -np.inf)
        take = min(k * OVERSAMPLE, scores.size)
        best = np.argpartition(-scores, kth=take - 1)[:take]
        best = best[np.argsort(-scores[best], kind="stable")]
        best = best[np.isfinite(scores[best])]
        if best.size == 0:
            return []

        # The relevance cutoff, before anything else looks at the shortlist: a candidate
        # further than PICKER_MARGIN below the best match is not an alternative reading of
        # the query, it is the fifth-best of whatever was left.
        keep = scores[best[0]] - (PICKER_MARGIN if margin is None else margin)
        best = best[scores[best] >= keep]

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

    def similar(self, isbn: str, k: int = 10, *, tau: float | None = None) -> list[Suggestion]:
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
        actually refuse it. Since M14.2 the two floors are separate numbers, because raising
        the candidate floor buys evidence and pays for it in relevance — see
        :attr:`DemoAssets.anchor_min_support`.

        Returns k suggestions whenever k exist. The score-gap truncation measured in M14.5
        is **off** (:data:`SCORE_TRUNCATION_TAU` is 0.0, the reversal); pass an
        explicit ``tau`` to apply it.
        """
        anchor = self._index.get(isbn)
        if anchor is None or self.assets.item_support[anchor] < self.assets.anchor_floor:
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
        # **Both sides of every comparison below come from `describe`**, which is what makes
        # them comparable: it resolves the dataset's HTML entities, and reading the anchor
        # straight off `assets.books` instead would compare an unescaped candidate author
        # against a raw `Anne Rice &amp; co` — the M14.3 bug class returning by a different
        # door. Unreachable today, because every work with an entity in its author string
        # sits below both floors, but a floor is a decision and decisions move.
        anchor_meta = self.describe(isbn)
        anchor_title = split_series(anchor_meta.title)[0]
        anchor_readers = self._readers_of(anchor)

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
                same_author=same_author(book.author, anchor_meta.author),
                shared_series=book.series if book.series and book.series == anchor_meta.series else "",
            )
            out.append(
                Suggestion(
                    isbn=book.isbn,
                    title=book.title,
                    author=book.author,
                    year=book.year,
                    series=book.series,
                    evidence=evidence,
                    reason=reason_sentence(evidence, anchor_title),
                )
            )
            if len(out) == k:
                break
        return _truncate(out, SCORE_TRUNCATION_TAU if tau is None else tau)

    def _readers_of(self, item_row: int) -> np.ndarray:
        readers = self.assets.readers
        lo, hi = readers.indptr[item_row], readers.indptr[item_row + 1]
        return readers.indices[lo:hi]
