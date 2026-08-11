"""The three neighbour engines the demo can be configured with (milestone M23).

**Why this module exists.** Until M23 the app had exactly one way to find neighbours — the
dot product of two ALS item factors — written inline in :meth:`DemoEngine.similar`. L81 and
L85 then found the same shape twice: every engine this project has measured is separated
from item-item **below** the app's anchor floor and indistinguishable **above** it. That
makes the floor and the engine one decision rather than two, and a decision cannot be
measured while one of its two arms is hard-wired.

**What a source is, and what it deliberately is not.** A source answers one question: *given
this anchor row, which rows are its best candidates and what number should be shown next to
each*. Everything else the app does with that answer — the candidate support floor, dropping
books it cannot name, work dedup, the co-reader count, the reason sentence — stays in
:meth:`DemoEngine.similar` and is shared by all three. That is deliberate and it is decision
6 of M23: *a switch changes the list and not the vocabulary*. A configuration that also
changed how evidence is counted would be two variables moving behind one control, which is
the defect this app has already had removed twice (M14.3, M17).

The three:

- :class:`AlsFactors` — **A**, what ships today. Cosine between L2-normalized ALS item
  factors (L34's support floor is applied by the caller, not here).
- :class:`ItemItemCosine` — **B**, the shrunk cosine of the published table's winner,
  ``co(i,j) / (sqrt(support_i · support_j) + λ)`` with λ = 10, computed from the same reader
  matrix the app already carries. Nothing is fitted at serving time; a similarity row is one
  sparse product.
- :class:`ReciprocalRankFusion` — **C**, L85's parameter-free winner: item-item and TF-IDF
  ranks fused by ``Σ 1/(60 + rank)``. It ranks through
  :func:`recommender.models.hybrid.combine_user_scored`, the same function that produced
  L85, so the configuration and the measurement cannot rank two different ways.

**One honest difference from L85, stated because it changes what C is.** In L85 the content
half was a bare ``TfidfRecommender`` with no support floor of its own. Here the app's
candidate floor (L34, 20 interactions) is applied to **both** halves before fusion, because
the app's standing contract is that it never shows a book that thin — and 23.1 is a question
about what a visitor sees. So this is the *serving* variant of RRF, and its numbers are not
interchangeable with L85's.

**Two of the three are configurations; all three stay measurable (M23.10).** The app offers
**A** and **B** through :data:`CONFIGURATIONS`. **C is dropped and does not come back**, and
the reason is written here rather than left to memory, because L85 ranks it *first* on the
item query and that number will tempt someone to re-add it: 17 of C's 80 audited slots are
text matches on a token of the anchor author's name (L89), so the rule that leads the metric
is also the only one that puts a book with **zero** shared readers in front of a reader.
:func:`build_source` keeps all three, because ``scripts/audit_floor_band.py`` and
``scripts/analyze_anchor_support.py`` are the commands L87, L88 and L89 were measured with and
a published measurement's command has to keep running. So: **the registry is what the app
offers, and** :func:`build_source` **is what the audit scripts measure.** Those are different
questions and conflating them is how a dropped configuration comes back by accident.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recommender.answers import load_answers
from recommender.models.content_tfidf import content_text
from recommender.models.hybrid import RRF, combine_user_scored

#: λ in the shrunk cosine. 10 is the value chosen on the inner validation split in M6 and
#: re-selected unchanged on the work-keyed matrix in M12 (L25, L51) — not re-tuned here,
#: because a serving surface that quietly retunes a published parameter is a number nobody
#: can trace.
DEFAULT_SHRINKAGE = 10.0

#: How deep each half of the fusion looks before the ranks are combined. 100 is
#: `hybrid.DEFAULT_CANDIDATES`, the depth L77/L78/L85 were all measured at.
DEFAULT_DEPTH = 100


class NeighbourSource(Protocol):
    """Ranked candidate rows for one anchor, plus the number to display beside each."""

    name: str

    #: What the displayed number means, for the label M17 requires the screen to carry.
    #: ``""`` means the source has no honest scalar to show — see :class:`ReciprocalRankFusion`.
    score_label: str

    def candidates(self, anchor: int, take: int) -> tuple[np.ndarray, np.ndarray]:
        """*take* rows ranked best-first and their scores, already masked and finite."""


def _rank(scores: np.ndarray, anchor: int, eligible: np.ndarray, take: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask, take the top *take*, sort stably. The shared tail of every dense source.

    Kept in one place because the stable sort is load-bearing: ties are common on this data
    and ``argpartition`` alone returns them in an order nobody could reproduce.
    """
    scores = np.asarray(scores, dtype=np.float64).copy()
    scores[~eligible] = -np.inf
    scores[anchor] = -np.inf
    take = max(1, min(take, scores.size - 1))
    best = np.argpartition(-scores, kth=take - 1)[:take]
    best = best[np.argsort(-scores[best], kind="stable")]
    finite = np.isfinite(scores[best])
    return best[finite], scores[best][finite]


class AlsFactors:
    """Configuration A: the engine the demo ships, extracted unchanged."""

    name = "als"
    score_label = "cosine between ALS item factors"

    def __init__(self, factors: np.ndarray) -> None:
        self.factors = factors

    def candidates(self, anchor: int, take: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(self.factors @ np.asarray(self.factors[anchor]))
        return _rank(scores, anchor, self._eligible, take)

    #: Set by :class:`DemoEngine`; a source never reads the floors itself, so the floor stays
    #: one decision taken in one place.
    _eligible: np.ndarray


class ItemItemCosine:
    """Configuration B: the published table's winner, served from the reader matrix.

    ``sim(i, j) = co(i, j) / (sqrt(support_i · support_j) + λ)`` — the formula in
    ``models/item_item.py``, evaluated for one row at a time.

    **One documented divergence from that model: its top-50 row truncation is not reproduced
    here.** For configuration B that is harmless — the top 10 of a full row and the top 10 of
    its top 50 are the same ten items. For configuration C it is **not**: RRF reads this list
    at depth 100 (:data:`DEFAULT_DEPTH`), so neighbours 51-100, which do not exist in the
    fitted model L85 measured, do reach the fusion and can move the fused top 10. C is
    therefore the *serving* variant of RRF rather than a re-run of L85's — the same caveat the
    module docstring already makes about the candidate floor, for a second reason. L87, L88
    and L89 were measured on this code, so the divergence is documented rather than removed:
    changing the ranking now would silently move three published lines.
    """

    name = "item-item"
    score_label = "shrunk cosine (λ=10) over shared readers"

    def __init__(self, readers: sp.csr_matrix, support: np.ndarray, shrinkage: float = DEFAULT_SHRINKAGE) -> None:
        self.readers = readers.tocsr()
        self.support = np.asarray(support, dtype=np.float64)
        self.shrinkage = shrinkage
        self._norm = np.sqrt(np.maximum(self.support, 0.0))

    def co_occurrence(self, anchor: int) -> np.ndarray:
        """Readers of *anchor* who also read each item — the app's own co-reader count."""
        return np.asarray((self.readers @ self.readers[anchor].T).todense(), dtype=np.float64).ravel()

    def candidates(self, anchor: int, take: int) -> tuple[np.ndarray, np.ndarray]:
        co = self.co_occurrence(anchor)
        scores = co / (self._norm * self._norm[anchor] + self.shrinkage)
        return _rank(scores, anchor, self._eligible, take)

    _eligible: np.ndarray


class TfidfText:
    """The content half of C: character n-gram TF-IDF over the app's own title+author text.

    Vectorized with the settings of :class:`recommender.models.content_tfidf.TfidfRecommender`
    and its ``content_text`` helper, so this is the same representation the published TF-IDF
    rows were measured on — applied to the app's work catalogue rather than the bench's.
    Items with no catalogue row get no vector and can never be a candidate, which is the app's
    existing rule (L46) rather than a new one.
    """

    name = "content-tfidf"
    score_label = "cosine between title+author character n-grams"

    def __init__(self, item_ids: np.ndarray, books, **kwargs) -> None:
        named = books.reindex(np.asarray(item_ids, dtype=object))
        self.has_text = named["Book-Title"].notna().to_numpy() & (named["Book-Title"].fillna("") != "").to_numpy()
        vectorizer = TfidfVectorizer(
            analyzer=kwargs.get("analyzer", "char_wb"),
            ngram_range=kwargs.get("ngram_range", (3, 5)),
            min_df=kwargs.get("min_df", 3),
            max_features=kwargs.get("max_features", 300_000),
            sublinear_tf=True,
            dtype=np.float32,
        )
        text = content_text(named.fillna(""))
        self.vectors = normalize(vectorizer.fit_transform(text))
        self.vocabulary = len(vectorizer.vocabulary_)

    def candidates(self, anchor: int, take: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.has_text[anchor]:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        scores = np.asarray((self.vectors @ self.vectors[anchor].T).todense(), dtype=np.float64).ravel()
        return _rank(scores, anchor, self._eligible & self.has_text, take)

    _eligible: np.ndarray


class ReciprocalRankFusion:
    """Configuration C: L85's parameter-free winner, ranked by the M22 function.

    ``Σ 1/(60 + rank)`` over the two halves' ranked lists. The number this returns is the
    fused rank sum, which is **not** a similarity — :attr:`score_label` is empty for exactly
    that reason, and M23 decision 7 requires the screen to print something honest or nothing
    rather than dress it up as one.
    """

    name = "rrf"
    score_label = ""

    def __init__(self, collaborative, content, *, depth: int = DEFAULT_DEPTH, rrf_k: int = 60) -> None:
        self.collaborative = collaborative
        self.content = content
        self.depth = depth
        self.rrf_k = rrf_k

    def candidates(self, anchor: int, take: int) -> tuple[np.ndarray, np.ndarray]:
        cf_rows, cf_scores = self.collaborative.candidates(anchor, self.depth)
        ct_rows, ct_scores = self.content.candidates(anchor, self.depth)
        if cf_rows.size == 0 and ct_rows.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        fused = combine_user_scored(
            cf_rows.astype(object),
            cf_scores,
            ct_rows.astype(object),
            ct_scores,
            rule=RRF,
            k=take,
            rrf_k=self.rrf_k,
        )
        rows = np.array([int(i) for i, _ in fused], dtype=np.int64)
        return rows, np.array([s for _, s in fused], dtype=np.float64)

    @property
    def _eligible(self) -> np.ndarray:
        return self.collaborative._eligible

    @_eligible.setter
    def _eligible(self, value: np.ndarray) -> None:
        self.collaborative._eligible = value
        self.content._eligible = value


def build_source(name: str, assets, **kwargs) -> NeighbourSource:
    """One of ``als`` / ``item-item`` / ``rrf`` over *assets*, with nothing fitted twice.

    The **measurement** entry point, and it keeps all three engines: L87, L88 and L89 were
    measured through it and a published line's command has to keep running. What the app
    offers is :data:`CONFIGURATIONS`, which is two.
    """
    if name == "als":
        return AlsFactors(np.asarray(assets.factors))
    if name == "item-item":
        return ItemItemCosine(assets.readers, assets.item_support, **kwargs)
    if name == "rrf":
        return ReciprocalRankFusion(
            ItemItemCosine(assets.readers, assets.item_support),
            TfidfText(assets.item_ids, assets.books),
        )
    raise ValueError(f"unknown engine {name!r}; expected one of als, item-item, rrf")


@dataclass(frozen=True)
class Configuration:
    """One position the demo can be switched to: an engine, and the words that go with it.

    The words are part of the configuration rather than of the app because of M23 decision 6:
    both configurations print a similarity, **both label which one it is**, and nothing on
    screen invites comparing the two numbers across a switch. A's is an ALS factor cosine and
    B's a shrunk cosine over shared readers; they are not comparable and the label is what
    stops the number being read across. Keeping the label beside the engine means a new
    configuration cannot arrive without one.
    """

    key: str
    #: The :func:`build_source` name this configuration runs.
    engine: str
    #: **What the picker shows**, and it is the model followed by the mechanism in three words
    #: (10.08.2026). The first draft labelled the two *Matrix factorization* and
    #: *Shared readers*, which named a model on one side and a mechanism on the other — and the
    #: accuracy table six lines above was by then marking "Item-based collaborative filtering ·
    #: this demo", so one screen carried two names for one thing. That is the M17.8/M17.10
    #: failure mode, twice removed from this app already.
    #:
    #: Written out, never a house abbreviation, per M15.5's register rule: a client does not
    #: know what "ALS" or "item-item" is. The gloss after the dash is what stops the model name
    #: being the only thing a non-specialist has to go on at the moment of choosing.
    label: str
    #: The same configuration as a **column heading or a table row** — the model, nothing else.
    #: A control label and an identifier have different jobs: the picker has to explain at the
    #: point of choice, a table row has to be recognisable at a glance and match its neighbours.
    #: The sidebar's accuracy table and ``docs/anchor_set_audit.md`` both read this field, so
    #: the row the "· this demo" marker lands on cannot drift away from the picker beside it.
    short_label: str
    #: Decision 6's label for the displayed number, in the legend under the list.
    #:
    #: **Plain language, and that is the register rule rather than a preference** (copy
    #: read, 10.08.2026). A's read "cosine between the books' learned profiles" for an
    #: afternoon, which put the one piece of undefined mathematics on this project's most
    #: client-facing screen — the same screen M15.5 cleared of ledger codes, and where three
    #: more captions were then cut for answering questions nobody had asked. B's label was
    #: already plain; A's now is. The *source* classes keep their technical labels, because
    #: those are read by the audit scripts and never by a visitor.
    score_label: str
    #: One sentence for the sidebar, in the same register as the rest of that copy.
    blurb: str
    #: Does this configuration answer from a precomputed table (M23.10.1)?
    #:
    #: **A does not, and that is decision 7 rather than an oversight.** A already ships the
    #: 155.6 MB factor matrix it answers from (L71); giving it a table would mean rebuilding
    #: ``artifacts/app/*``, which is the one thing this milestone may not do — the rehearsed
    #: eleven have to come back byte-identical, and the last rebuild moved the lookup between
    #: two screenshots (M17.9). B is additive: a new 0.3 MB file beside the old assets.
    table: bool = False


#: What the app offers, in picker order. **A is the default and stays the default** (M23
#: decision 2): if anything about B is unconvincing shortly before a demo it disappears
#: with one flag in ``app/main.py`` and A is untouched.
#:
#: **Both sit at anchor floor 50** (decision 1, changed 10.08.2026 on the review question
#: "can't both use the same threshold?"). At a common floor the switch moves **one** variable — the
#: same anchor, the same 2,508 askable works, answered from a calibrated similarity instead of
#: an anti-calibrated one (L87: 1.7% thin slots against 18.2%). The floor's own cost is a
#: separate demonstration with no picker in it: type *The Kite Runner* and the app declines,
#: under either configuration, because 39 readers is below 50.
CONFIGURATIONS: dict[str, Configuration] = {
    "A": Configuration(
        key="A",
        engine="als",
        label="Matrix factorization — learned profiles",
        short_label="Matrix factorization (ALS)",
        score_label="how closely two books' profiles line up",
        blurb=(
            "Every book gets a short profile learned from who read it. Two books are similar "
            "when their profiles point the same way."
        ),
    ),
    "B": Configuration(
        key="B",
        engine="item-item",
        label="Item-based collaborative filtering — shared readers",
        short_label="Item-based collaborative filtering",
        score_label="shared readers, weighted against how widely each book is read",
        blurb=(
            "Two books are similar when the same people read both. That count is weighed "
            "against how widely each book is read, so a bestseller does not come out "
            "similar to everything."
        ),
        table=True,
    ),
}

#: The one the demo runs unless somebody switches it.
DEFAULT_CONFIGURATION = "A"


def configuration(key: str | Configuration) -> Configuration:
    """Look up a configuration by key, or pass one through. Unknown keys are an error."""
    if isinstance(key, Configuration):
        return key
    try:
        return CONFIGURATIONS[key]
    except KeyError:
        raise ValueError(
            f"unknown configuration {key!r}; expected one of {', '.join(CONFIGURATIONS)}"
        ) from None


def build_configuration(
    key: str | Configuration,
    assets,
    *,
    table_dir: Path | None = None,
    live: bool = False,
) -> NeighbourSource:
    """The source for one configuration, from its answer table where one exists.

    **The table and the live source are the same ranking, and that is asserted rather than
    claimed** — ``scripts/build_answer_table.py`` writes the table by asking the live source,
    then re-reads it and compares every stored slot back against a live recomputation, and
    reports the worst score deviation it found. So "table or live" is a provenance question,
    not a behaviour question, which is what makes the fallback below safe.

    The fallback exists for one operational reason: a missing or stale table shortly before a
    demo must not be able to take the demo down. It **warns** rather than falling back
    quietly — the live source is the truth the table is a cache of, so answering from it is
    correct, but a serving path that silently stops using its asset is how a stale cache
    survives a week. ``live=True`` forces the computation, which is what the build script and
    the audit use.
    """
    config = configuration(key)
    directory = table_dir or getattr(assets, "directory", None)
    if config.table and not live and directory is not None:
        try:
            return load_answers(directory, config.engine, assets)
        except (FileNotFoundError, RuntimeError) as error:
            warnings.warn(
                f"configuration {config.key} is answering from a live computation: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
    return build_source(config.engine, assets)
