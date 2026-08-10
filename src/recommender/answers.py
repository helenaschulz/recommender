"""The demo's precomputed answer table: one engine's top-10 for every askable work (M23.10).

**Serving, not evaluation.** Every row in this table is fitted on the *full* work-keyed
interaction matrix, exactly as ``scripts/build_app_assets.py`` fits the ALS factors and for
the same reason: the app computes no metric, so withholding a reader's interactions would
only make the product worse. Nothing measured on a train/test split may be read off this
file, and no published number is.

**Why a table exists at all, stated honestly rather than as a latency claim.** Configuration
B's similarity is one sparse product against the reader matrix the app already carries — 9 ms
per anchor on this machine, which is *faster* than the ALS factor path it sits beside (44 ms,
measured in the build script's own report, ledger L91). So this table does not buy speed and
it is not offered as buying speed. What it buys is the thing ``app/main.py``'s docstring has
argued for since M16 and never had: the answers a visitor sees are **built once and pinned**,
the way a Gold table is pinned in the Part 3 architecture, instead of being recomputed inside
the request. It is 0.3 MB against the 890 MB of assets it sits next to (L71, L72).

**Additive, and that is a hard constraint (M23 decision 7).** Nothing here rewrites, reads
back or invalidates the arrays ``build_app_assets.py`` wrote. The table is a new file beside
them, keyed on the **work key already stamped in ``meta.json``** — so configuration A's
answers cannot move because this file exists, and the eleven rehearsed anchors are byte for
byte what they were (verified in ``scripts/verify_configuration_a.py``).

**What "askable" means here is L65's word and L65's filter**, unchanged: a work with at least
``anchor_min_support`` interactions **that the catalogue can name**. That is 2,508 works, not
the 2,532 above the floor — the 24 difference are ids with no catalogue row, which no visitor
can type and which the app would refuse to print anyway (L46). *(L72 first published 2,532 and
was corrected for exactly this reason; the same filter is applied here so the two agree.)*

**Staleness is a loud failure, not a silent one.** A table built against one set of assets and
read against another would answer with the right numbers attached to the wrong books, which is
the worst failure mode available to a serving cache. :func:`load_answers` therefore refuses to
return a table whose stamp does not match the assets in front of it — and the stamp includes a
hash of the support vector, so it catches a *rebuild that kept the shape* and not only one that
changed it. The hash is over ``item_support`` (2.4 MB, ~5 ms) rather than over the 328 MB id
column, because the check runs on the demo's cold-start path and L69's 10.6 s is a budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Bumped when the on-disk layout below changes, so an old table fails with a sentence
#: rather than with a shape error inside a query. Same rule as ``demo.ASSET_VERSION``.
TABLE_VERSION = 1

#: One file per engine, beside the assets rather than inside them.
FILENAME = "answers_{engine}.npz"


def table_path(directory: Path, engine: str) -> Path:
    return Path(directory) / FILENAME.format(engine=engine)


def askable_rows(assets) -> np.ndarray:
    """L65's filter, unchanged: above the anchor floor **and** nameable by the catalogue.

    The one definition of what a table covers, so the builder, the audit and anything that
    quotes a count cannot drift apart about it. The nameable half is not a detail: 2,532 rows
    clear the floor and 24 of them have no catalogue row, so the app can neither offer them nor
    print them (L46). "Askable" is L65's word for the 2,508 that are left, and L72 was
    corrected on exactly this point — a table that covered 2,532 would disagree with every
    published count of the same set.
    """
    named = set(assets.books.index)
    nameable = np.array([isbn in named for isbn in assets.item_ids.tolist()])
    return np.flatnonzero(nameable & (assets.item_support >= assets.anchor_floor))


def assets_stamp(assets) -> dict[str, object]:
    """What a table must agree with about the assets it was built from.

    Four fields, each of which has actually moved in this project's history: the item count
    (M12's re-key), the work key itself (L64's punctuation fix moved 0.5% of works and with
    them a third of every neighbourhood, L67), and the two floors (M14.2 split them apart).
    The support hash is the backstop for the case none of the four can see — a rebuild that
    lands on the same shape with different contents.
    """
    support = np.ascontiguousarray(assets.item_support)
    return {
        "n_items": int(len(assets.item_ids)),
        "work_key": str(getattr(assets, "work_key", "") or ""),
        "similar_min_support": int(assets.similar_min_support),
        "anchor_min_support": int(assets.anchor_floor),
        "item_support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
    }


#: Not frozen, and for one reason: :class:`~recommender.demo.DemoEngine` sets ``_eligible`` on
#: whatever source it is handed, because the candidate floor is the engine's decision and not
#: the source's (M23 decision 6). A frozen source would make a table the one kind of source
#: that could not be told about a floor.
@dataclass
class AnswerTable:
    """One engine's top-10 per askable anchor, as a neighbour source.

    Implements the :class:`recommender.engines.NeighbourSource` protocol, so
    :meth:`recommender.demo.DemoEngine.similar` cannot tell it apart from a live source —
    which is the property that makes M23 decision 6 hold through a cache: *a switch changes
    the list and not the vocabulary*. Dedup, the co-reader count, the thin-evidence tag and
    the reason sentence are all still computed by the engine, from the assets, at query time.
    Only the ranking is remembered.

    **The rows stored are the ones the engine kept**, i.e. after the nameability filter of
    L46, not the raw top-100 the live source returns. That is what makes the table 25,080
    rows rather than 250,800, and it is why :meth:`candidates` ignores *take* above ten: the
    depth was spent at build time. ``similar(k=…)`` above 10 therefore returns at most ten
    slots under a table-backed configuration, which the app never asks for and which the
    build script's ``--k`` records.
    """

    engine: str
    score_label: str
    #: Anchor row -> slice into :attr:`rows` / :attr:`scores`.
    index: dict[int, tuple[int, int]]
    rows: np.ndarray  # int32
    scores: np.ndarray  # float32
    meta: dict[str, object]

    #: Set by :class:`~recommender.demo.DemoEngine`, exactly as on a live source. A table is
    #: built under one candidate floor and stamped with it, so this mask normally changes
    #: nothing — it is applied rather than assumed because the floor is the *engine's*
    #: decision (M23 decision 6) and a source that quietly ignored it would be the one place
    #: switching configurations could move a floor.
    _eligible: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.engine

    @property
    def provenance(self) -> str:
        return f"answer table, {self.meta.get('built_at', 'unknown date')}"

    def candidates(self, anchor: int, take: int) -> tuple[np.ndarray, np.ndarray]:
        span = self.index.get(int(anchor))
        if span is None:
            # Above the floor but not in the table: an id with no catalogue row (24 of them).
            # The engine would refuse to print it anyway; saying nothing is the honest answer
            # rather than falling back to a computation this table exists to replace.
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        lo, hi = span
        rows = self.rows[lo:hi].astype(np.int64)
        scores = self.scores[lo:hi].astype(np.float64)
        if self._eligible is not None:
            keep = self._eligible[rows]
            rows, scores = rows[keep], scores[keep]
        return rows[:take], scores[:take]


def save_answers(
    path: Path,
    *,
    engine: str,
    score_label: str,
    anchors: np.ndarray,
    indptr: np.ndarray,
    rows: np.ndarray,
    scores: np.ndarray,
    assets,
    meta: dict[str, object],
) -> Path:
    """Write one engine's table. ``int32`` ids and ``float32`` scores, L72's 12 bytes a row.

    ``float32`` is the same convention L72 priced the table at and is four decimal digits
    wider than anything the app prints (two on screen, three in the anchor report). The build
    script measures the round-trip deviation against the live source rather than asserting it
    is small, and records the number.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = dict(assets_stamp(assets), table_version=TABLE_VERSION, engine=engine, score_label=score_label)
    np.savez(
        path,
        anchors=np.asarray(anchors, dtype=np.int32),
        indptr=np.asarray(indptr, dtype=np.int32),
        rows=np.asarray(rows, dtype=np.int32),
        scores=np.asarray(scores, dtype=np.float32),
        meta=np.array(json.dumps(dict(stamp, **meta)), dtype=object),
    )
    return path


def load_answers(directory: Path, engine: str, assets) -> AnswerTable:
    """Read *engine*'s table, or raise with the command that rebuilds it.

    Raises :class:`FileNotFoundError` when there is no table and :class:`RuntimeError` when
    there is one that does not belong to these assets. Callers that can live without it —
    the app, which simply does not offer the configuration — catch both.
    """
    path = table_path(directory, engine)
    if not path.exists():
        raise FileNotFoundError(
            f"no answer table for {engine!r} at {path}. Build it with:\n"
            f"    python scripts/build_answer_table.py --engine {engine}"
        )
    with np.load(path, allow_pickle=True) as handle:
        meta = json.loads(str(handle["meta"].item()))
        anchors = handle["anchors"]
        indptr = handle["indptr"]
        rows = handle["rows"]
        scores = handle["scores"]

    if int(meta.get("table_version", -1)) != TABLE_VERSION:
        raise RuntimeError(
            f"{path} is table version {meta.get('table_version')}, this code expects "
            f"{TABLE_VERSION}. Re-run: python scripts/build_answer_table.py --engine {engine}"
        )
    stamp = assets_stamp(assets)
    stale = {key: (meta.get(key), value) for key, value in stamp.items() if meta.get(key) != value}
    if stale:
        detail = "; ".join(f"{key}: table {was!r}, assets {now!r}" for key, (was, now) in stale.items())
        raise RuntimeError(
            f"{path} was built against different assets ({detail}). Re-run: "
            f"python scripts/build_answer_table.py --engine {engine}"
        )

    index = {int(a): (int(indptr[i]), int(indptr[i + 1])) for i, a in enumerate(anchors.tolist())}
    return AnswerTable(
        engine=engine,
        score_label=str(meta.get("score_label", "")),
        index=index,
        rows=rows,
        scores=scores,
        meta=meta,
    )
