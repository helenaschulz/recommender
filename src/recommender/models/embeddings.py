"""Multilingual sentence embeddings over title + author — the content coverage layer.

This is the deep-learning component of the project, and it is deliberately a *feature
extractor* rather than a trained scoring model: a pretrained transformer turns each
book's title and author into a vector, similarity is cosine, and coverage of the
catalogue is 100% from day one. The evidence on Book-Crossing says simple models win in
the scoring core (Naghiaei et al. 2022, https://arxiv.org/abs/2202.13446);
it says nothing against using a transformer where it is genuinely better than the
alternative, which is exactly here.

**Why embeddings rather than the TF-IDF layer alone.** The catalogue is multilingual —
German, French and Spanish titles sit beside English ones — so a lexical model can only
relate *Der Herr der Ringe* to *The Lord of the Rings* through characters they happen to
share. A multilingual sentence encoder places them near each other because it was
trained to. The honest limit is the same for both: title and author are a handful of
words, which is thin evidence about a book. That thinness is precisely what an LLM
metadata-enrichment layer would fix — generating genre tags, themes and a short
description per book — which is why that layer is proposed rather than assumed.

**Profile centering, chosen on validation.** Sentence embeddings share a large common
component — every vector points partly the same way — so averaging a user's books yields
almost the same profile vector for every user (measured: mean cosine to the global
profile centroid **0.883**, against 0.518 for the item vectors themselves). The model
then recommends the same generic region to everybody. Subtracting the global mean from
the item vectors and renormalizing removes that shared direction. On the inner validation
split (seed 43, never the test holdout) it lifts HitRate@10 from 0.0036 to **0.0095** and
Coverage@10 from 3.7% to **20.4%**, and drops profile collapse from 0.893 to 0.193. It is
on by default; ``center=False`` reproduces the uncentered ablation.

**Two products from one artefact.** The same vectors serve the recommender *and* the
app's input path: :meth:`EmbeddingRecommender.find_book` turns free text like
``"harry potter stein"`` into a catalogue entry without an exact string match. On
Databricks both are one Mosaic AI Vector Search index.

**Caching.** Encoding the catalogue costs minutes, so vectors are written to a
gitignored ``artifacts/`` directory keyed by model name and catalogue fingerprint, and
reused on every later run. Tests never touch any of this: they inject a fake encoder, so
no test downloads a model or reaches the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from recommender.data import BookCrossing, Interactions, project_root
from recommender.models.base import Recommender, top_k_from_scores
from recommender.models.content_tfidf import content_text

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE = Path("artifacts/embeddings")  # resolved against the project root, not the cwd

Encoder = Callable[[list[str]], np.ndarray]


def _fingerprint(model_name: str, texts: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(str(len(texts)).encode())
    for text in texts[:: max(1, len(texts) // 512)]:
        digest.update(str(text).encode())
    return digest.hexdigest()[:16]


def sentence_transformer_encoder(model_name: str = DEFAULT_MODEL, batch_size: int = 256) -> Encoder:
    """Build an encoder backed by sentence-transformers. Downloads the model on first use.

    The loaded model is held in the closure, so encoding the catalogue and then answering
    a hundred free-text lookups costs one load, not a hundred. That matters for the app:
    re-instantiating a transformer per query would put seconds of latency on every
    keystroke-driven search.
    """
    holder: dict[str, object] = {}

    def encode(texts: list[str]) -> np.ndarray:
        import torch
        from sentence_transformers import SentenceTransformer

        if "model" not in holder:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            holder["model"] = SentenceTransformer(model_name, device=device)
        model = holder["model"]
        return np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    return encode


class EmbeddingRecommender(Recommender):
    """Cosine similarity over multilingual sentence embeddings of title + author."""

    name = "content-embeddings"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        encoder: Encoder | None = None,
        cache_dir: Path | str | None = DEFAULT_CACHE,
        batch_size: int = 256,
        score_batch: int = 256,
        center: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if name:
            self.name = name
        self.model_name = model_name
        self.encoder = encoder
        self.cache_dir = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            self.cache_dir = cache_dir if cache_dir.is_absolute() else project_root() / cache_dir
        self.batch_size = batch_size
        self.score_batch = score_batch
        self.center = center
        self.params = {
            "model": model_name,
            "text": "title + author, lower-cased",
            "similarity": "cosine on L2-normalized embeddings",
            "scoring": "mean cosine to the user's train items",
            "centered": center,
        }
        self.vectors: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.cache_path: Path | None = None
        self.lookup_vectors: np.ndarray | None = None
        self._centroid: np.ndarray | None = None
        self._index: dict[str, int] = {}

    def fit(self, train: Interactions, catalog: BookCrossing) -> EmbeddingRecommender:
        self.train = train
        books = catalog.books.loc[catalog.books["Book-Title"].notna()]
        self.item_ids = books["ISBN"].to_numpy()
        self._index = {isbn: i for i, isbn in enumerate(self.item_ids)}
        texts = content_text(books)

        self.vectors = self._load_or_encode(texts)
        # Free-text lookup is served from the *uncentered* vectors, deliberately: see
        # find_book. Kept as a separate array because normalizing after centering is
        # lossy, so the original cannot be recovered from self.vectors.
        self.lookup_vectors = self.vectors
        if self.center:
            # Sentence embeddings share a large common component: every vector points
            # partly in the same direction. Averaging a user's books therefore produces
            # nearly the same profile vector for everyone (measured: mean cosine to the
            # global profile centroid 0.883, against 0.518 for the item vectors) and the
            # model recommends the same generic region to all of them. Subtracting the
            # global mean and renormalizing removes that shared direction -- the standard
            # centering fix for dense retrieval.
            self._centroid = self.vectors.mean(axis=0, keepdims=True)
            self.vectors = self.vectors - self._centroid
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors = self.vectors / norms
        self.params["catalogue_embedded"] = len(self.item_ids)
        self.params["dimensions"] = int(self.vectors.shape[1])
        if self.cache_path is not None:
            self.params["vector_cache"] = str(self.cache_path)
        return self

    def _load_or_encode(self, texts: np.ndarray) -> np.ndarray:
        if self.cache_dir is not None:
            slug = self.model_name.rsplit("/", 1)[-1]
            self.cache_path = self.cache_dir / f"{slug}-{_fingerprint(self.model_name, texts)}.npy"
            if self.cache_path.exists():
                return np.load(self.cache_path)

        encoder = self.encoder or sentence_transformer_encoder(self.model_name, self.batch_size)
        vectors = np.asarray(encoder([str(t) for t in texts]), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, vectors)
        return vectors

    def _profile(self, user_row: int) -> np.ndarray | None:
        train = self._require_fit()
        lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
        isbns = train.item_ids[train.matrix.indices[lo:hi]]
        rows = [self._index[isbn] for isbn in isbns.tolist() if isbn in self._index]
        if not rows:
            return None
        return self.vectors[rows].mean(axis=0)

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        train = self._require_fit()
        out = np.full((len(user_ids), k), None, dtype=object)
        profiles, targets, blocked = [], [], []
        for row, user_id in enumerate(user_ids):
            user_row = train.user_index.get(int(user_id))
            if user_row is None:
                continue
            profile = self._profile(user_row)
            if profile is None:
                continue
            lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
            seen = train.item_ids[train.matrix.indices[lo:hi]]
            profiles.append(profile)
            targets.append(row)
            blocked.append(np.array([self._index[i] for i in seen.tolist() if i in self._index], dtype=np.int64))

        for start in range(0, len(profiles), self.score_batch):
            chunk = np.vstack(profiles[start : start + self.score_batch])
            scores = chunk @ self.vectors.T
            picked = top_k_from_scores(scores, k, blocked=blocked[start : start + self.score_batch])
            for local, row in enumerate(targets[start : start + self.score_batch]):
                chosen = picked[local]
                usable = chosen >= 0
                out[row, : usable.sum()] = self.item_ids[chosen[usable]]
        return out

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        self._require_fit()
        item = self._index.get(isbn)
        if item is None:
            return []
        return self._top(self.vectors[item], k, exclude=item)

    def find_book(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Free text in, catalogue entries out — the app's input path.

        ``"harry potter stein"`` should find the book without an exact string match, and
        without the user having to know the ISBN or the exact edition title.

        **This path uses the uncentered vectors, and the two product paths genuinely want
        different geometry.** Centering helps recommendation by removing the shared
        direction that makes every user's averaged profile look alike; but a lookup query
        *is* a single point, not an average, and the common direction is part of what
        matches it to a title. Measured over seven queries, centering pushed the correct
        book from rank 1 to 4 (``"el senor de los anillos"``), 2 to 5
        (``"harry potter stein"``) and 3 to 4 (``"da vinci code"``), with the same 5/7
        found in the top five.
        """
        self._require_fit()
        encoder = self.encoder or sentence_transformer_encoder(self.model_name, self.batch_size)
        vector = np.asarray(encoder([query.lower().strip()]), dtype=np.float32).ravel()
        norm = np.linalg.norm(vector)
        return self._top(vector / (norm or 1.0), k, vectors=self.lookup_vectors)

    def _top(
        self, vector: np.ndarray, k: int, exclude: int | None = None, vectors: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        scores = (self.vectors if vectors is None else vectors) @ vector
        if exclude is not None:
            scores[exclude] = -np.inf
        take = min(k, scores.size - (1 if exclude is not None else 0))
        best = np.argpartition(-scores, kth=take - 1)[:take]
        best = best[np.argsort(-scores[best], kind="stable")]
        return [(self.item_ids[i], float(scores[i])) for i in best if np.isfinite(scores[i])]
