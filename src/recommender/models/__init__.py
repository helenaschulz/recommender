"""Candidate models. One module per model, all behind the :class:`Recommender` interface.

The ladder, in the order the milestones build it:

- :mod:`~recommender.models.popularity` — the reference row. Everything else is only
  interpretable against it.
- :mod:`~recommender.models.item_item` — the core hypothesis: shrunk cosine over the
  binarized interaction matrix.
- :mod:`~recommender.models.content_tfidf` — the coverage layer over title+author.
- :mod:`~recommender.models.als` — the same story one rung up: item factors are
  embeddings, similarity is a dot product, and it carries to Spark for productionization.
- :mod:`~recommender.models.embeddings` — multilingual sentence embeddings, coverage of
  the whole catalogue from day one.
"""

import pandas as pd

from recommender.data import BookCrossing, Interactions, build_interactions
from recommender.models.base import Recommender, top_k_from_scores

__all__ = ["ALL_MODELS", "REGISTRY", "Recommender", "build_model", "fit_model", "top_k_from_scores"]

#: name -> (module, class, kwargs, the milestone that builds it). Deliberately complete:
#: it is the whole ladder, so a rung that does not exist yet fails with a sentence saying
#: which milestone builds it rather than an ImportError.
REGISTRY: dict[str, tuple[str, str, dict, str]] = {
    "popularity": ("popularity", "PopularityRecommender", {}, "M5"),
    "item-item": ("item_item", "ItemItemRecommender", {}, "M6"),
    "item-item-explicit": (
        "item_item",
        "ItemItemRecommender",
        {"signal": "explicit", "name": "item-item (explicit-only)"},
        "M6",
    ),
    "tfidf": ("content_tfidf", "TfidfRecommender", {}, "M7"),
    "als": ("als", "ALSRecommender", {}, "M8"),
    "embeddings": ("embeddings", "EmbeddingRecommender", {}, "M9"),
}
ALL_MODELS = list(REGISTRY)

#: Hyperparameters that differ once the model is fitted on the **work-keyed** matrix
#: (milestone M12.2), each chosen on the work-level inner validation split (seed 43) by
#: ``scripts/tune_item_item.py --work-level`` and ``scripts/tune_als.py --work-level``.
#:
#: A model absent from this mapping is not an omission and not an assumption: it means the
#: work-level sweep was run and the ISBN-level values won it again. Both outcomes are
#: recorded in the ledger, because "we re-tuned and nothing moved" is a measurement and
#: "we forgot to re-tune" looks identical from the outside.
WORK_LEVEL_PARAMS: dict[str, dict] = {}


def build_model(name: str, *, work_level: bool = False) -> Recommender:
    """Construct an unfitted model by its registry name, importing it lazily.

    Lazy so that a model whose optional dependency is missing breaks only itself.

    Args:
        work_level: use the parameter set validated on the work-keyed matrix
            (:data:`WORK_LEVEL_PARAMS`) instead of the ISBN-level one. The two published
            tables are measured at different item levels, so they are entitled to
            different hyperparameters — but only ones chosen on their own validation
            split, and only ones written down here.
    """
    if name not in REGISTRY:
        raise SystemExit(f"unknown model {name!r}. Known: {', '.join(ALL_MODELS)}")
    module_name, class_name, kwargs, milestone = REGISTRY[name]
    if work_level:
        kwargs = {**kwargs, **WORK_LEVEL_PARAMS.get(name, {})}
    try:
        module = __import__(f"recommender.models.{module_name}", fromlist=[class_name])
    except ImportError as exc:
        raise SystemExit(f"model {name!r} is not available yet (built in milestone {milestone}): {exc}") from exc
    return getattr(module, class_name)(**kwargs)


def fit_model(
    model: Recommender,
    train: Interactions,
    catalog: BookCrossing,
    train_ratings: pd.DataFrame,
) -> Recommender:
    """Fit *model* on the right view of the training data.

    Two models want something other than the plain binarized matrix, and this is the one
    place that knows it. It exists because it once did not: the runner special-cased both
    while the notebook did not, so the notebook silently reported the explicit-only
    ablation as an exact duplicate of the binarized run. Any dispatch worth writing twice
    is worth writing once.

    - ``signal="explicit"`` (the item-item ablation) is fitted on a matrix built from
      graded interactions alone, over the *same* item and user index space so the two
      models stay comparable coordinate for coordinate.
    - ALS additionally needs the train ratings frame for its ``1 + alpha * rating``
      confidence weights. It is passed explicitly and never read from ``catalog.ratings``,
      which still contains the holdout.
    """
    fit_matrix = train
    if getattr(model, "signal", "binary") == "explicit":
        fit_matrix = build_interactions(
            train_ratings[train_ratings["is_explicit"]],
            weights="binary",
            item_ids=train.item_ids,
            user_ids=train.user_ids,
        )

    if type(model).__name__ == "ALSRecommender":
        return model.fit(fit_matrix, catalog, ratings=train_ratings)
    return model.fit(fit_matrix, catalog)
