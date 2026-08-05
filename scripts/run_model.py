"""Run one or more models on the pinned split and print ledger-ready numbers.

    python scripts/run_model.py popularity item-item
    python scripts/run_model.py --all --gallery

Thin wrapper: it wires data -> split -> model -> eval and prints. No scoring logic lives
here, so nothing measured through this script can differ from what the notebook measures
through the same package.
"""

from __future__ import annotations

import argparse
import sys
import time

from recommender.data import build_interactions, load
from recommender.eval import comparison_table, evaluate
from recommender.gallery import build_gallery, render_markdown
from recommender.models import fit_model
from recommender.split import make_split

# name -> (module, class, kwargs, milestone that builds it).
# The registry is deliberately complete: it is the full model ladder, and a model that is
# not built yet fails with a message saying so rather than an ImportError.
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


def build_model(name: str):
    """Import lazily, so a model that is not built yet (or whose optional dependency is
    missing) only breaks itself, never the whole run."""
    if name not in REGISTRY:
        raise SystemExit(f"unknown model {name!r}. Known: {', '.join(ALL_MODELS)}")
    module_name, class_name, kwargs, milestone = REGISTRY[name]
    try:
        module = __import__(f"recommender.models.{module_name}", fromlist=[class_name])
    except ImportError as exc:
        raise SystemExit(
            f"model {name!r} is not available yet (built in milestone {milestone}): {exc}"
        ) from exc
    return getattr(module, class_name)(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help=f"run {', '.join(ALL_MODELS)}")
    parser.add_argument("--gallery", action="store_true", help="also print the 3-anchor gallery")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit-users", type=int, default=None, help="runtime guardrail: evaluate on a seeded sample")
    args = parser.parse_args(argv)

    names = ALL_MODELS if args.all else args.models
    if not names:
        parser.error("name at least one model, or pass --all")

    started = time.perf_counter()
    catalog = load()
    split = make_split(catalog.ratings)
    train = build_interactions(split.train, weights="binary")
    catalog_isbns = set(catalog.books["ISBN"])
    print(split.describe())
    print(f"data ready in {time.perf_counter() - started:.0f}s\n", flush=True)

    users = None
    if args.limit_users:
        import numpy as np

        rng = np.random.default_rng(split.seed)
        eligible = split.test["User-ID"].to_numpy()
        users = rng.choice(eligible, size=min(args.limit_users, len(eligible)), replace=False)
        print(f"RUNTIME GUARDRAIL: evaluating on a seeded sample of {len(users):,} users\n")

    results = []
    fitted = []
    for name in names:
        model = build_model(name)
        t0 = time.perf_counter()
        # One dispatch, shared with the notebook: see recommender.models.fit_model.
        fit_model(model, train, catalog, split.train)
        fit_seconds = time.perf_counter() - t0
        result = evaluate(
            model,
            split,
            train,
            catalog_isbns=catalog_isbns,
            k=args.k,
            users=users,
            notes=f"evaluated on a seeded sample of {len(users):,} users" if users is not None else "",
        )
        print(f"{result}   [fit {fit_seconds:.0f}s]  params: {result.params}", flush=True)
        results.append(result)
        fitted.append(model)

    print("\n" + comparison_table(results).to_markdown(index=False))

    if args.gallery:
        print("\n" + render_markdown(build_gallery(fitted, catalog, k=args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
