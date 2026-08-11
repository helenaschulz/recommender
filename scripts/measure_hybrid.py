"""Measure the hybrid the project has been recommending since M10 (milestone M19).

    python scripts/measure_hybrid.py                 # the three rules on the pinned split
    python scripts/measure_hybrid.py --tune          # + the alpha curve on inner validation
    python scripts/measure_hybrid.py --gallery       # + the three-anchor gallery
    python scripts/measure_hybrid.py --pairs-from artifacts/hybrid/rules_work_k10.npz

**The prediction, written down before the run and repeated here so it cannot be quietly
edited afterwards** (09.08.2026, from L20/L50 and L27/L73):

    Coverage will move a lot and HitRate will barely move, possibly down. The held-out
    items only the content layer can reach are by definition items with little or no
    interaction evidence, and no model scores meaningfully in that stratum. So the 8.7
    points of extra *reachable* ceiling are reachable in principle and mostly unrankable
    in practice. If HitRate rises at all it should be by a little; if it rises by a lot,
    treat it as a bug first and look for the content model being handed items the
    collaborative model had already ranked.

L73 sharpens one clause of that: at zero train support TF-IDF is **not** 0.0000, it is
0.0304 — small, but the only nonzero cell in that stratum. So there is something for a
cascade to win, and the question is whether ten slots are worth spending on it.

**What this script does, in the order that matters.**

1. **Reproduce first, combine second.** Item-item and TF-IDF are fitted and scored exactly
   as the comparison table scores them, and all three of each model's published metrics must
   come back to the digit. Their per-user hit vectors are additionally compared **bit for
   bit** against the vectors M18 cached, because M19's escalation rule is "if the cascade
   changes item-item's own rows by any amount, the cascade is leaking into the base model"
   — and four decimals of HitRate is a weaker check than 13,580 booleans.
2. **Score the candidates once.** Both models are asked for their top-100 per user a single
   time; all three rules then run over the same arrays. This is why three rules cost barely
   more than one, and it is also why the rules are exactly comparable: identical inputs.
3. **Report the honesty column.** For every rule: how many of its hits came from slots
   item-item would have filled anyway. A hybrid whose hits are all item-item's hits is
   item-item with extra steps, and that column is what makes the row worth having.

**Tuning never sees the holdout.** ``--tune`` carves the inner validation split out of train
(seed 43, :func:`recommender.benchmark.inner_bench`, the same harness L51 used), re-fits both
models there, and sweeps alpha on that. The whole curve is reported, not the argmax.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
import pandas as pd

from recommender.benchmark import Bench, build_bench, inner_bench
from recommender.eval import catalog_coverage_at_k, hit_vector, mcnemar, novelty_at_k, wilson_interval
from recommender.gallery import build_gallery, render_markdown
from recommender.models import build_model, fit_model
from recommender.models.hybrid import CASCADE, DEFAULT_CANDIDATES, FUSION, RRF, HybridRecommender, combine_user

#: The published work-level rows of the two models being combined (L53, L56), as
#: (HitRate@10, Coverage@10 %, Novelty@10). The run stops if either has moved.
PUBLISHED = {
    "item-item": (0.0644, 8.190, 14.17),
    "tfidf": (0.0405, 16.806, 17.07),
}

#: M18's cached per-user hit vectors. Present after `scripts/measure_significance.py`; when
#: absent the bit-for-bit check is skipped and says so, rather than silently passing.
HIT_CACHE = "artifacts/significance/hits_work_k10.npz"

ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def pairwise_report(path: str, base_cache: str) -> int:
    """Paired McNemar between every pair of saved rules — no re-fit, no re-scoring.

    ``--save`` writes each rule\'s top-k array; this reads them back and answers the
    questions the main table cannot, because they are comparisons *between* rules rather
    than against the base model. The one that decides the milestone is tuned fusion against
    parameter-free RRF: if the tuning cannot be told apart from the constant, the tuning
    bought nothing, and M19 said in advance that this goes in the line either way.
    """
    stored = np.load(path, allow_pickle=True)
    holdout = stored["holdout"]
    hits: dict[str, np.ndarray] = {}
    for name in stored.files:
        if name in ("holdout", "users"):
            continue
        array = stored[name].astype(object)
        array[array == "None"] = None
        hits[name] = hit_vector(array, holdout)
    try:
        hits["item-item (published)"] = np.load(base_cache, allow_pickle=False)["item-item"].astype(bool)
    except FileNotFoundError:
        print(f"note: {base_cache} absent — the published base row is not in this comparison")

    n = len(holdout)
    print(
        pd.DataFrame(
            [
                {
                    "rule": name,
                    "HitRate@10": round(float(v.mean()), 5),
                    "hits": int(v.sum()),
                    "95% CI (Wilson)": "[{:.4f}, {:.4f}]".format(*wilson_interval(int(v.sum()), n)),
                }
                for name, v in hits.items()
            ]
        ).to_markdown(index=False)
    )

    names = list(hits)
    rows = []
    for a, b in itertools.combinations(names, 2):
        test = mcnemar(hits[a], hits[b])
        rows.append(
            {
                "A": a,
                "B": b,
                "A only": test["a_only"],
                "B only": test["b_only"],
                "Δ HitRate": f"{test['diff']:+.4f}",
                "95% CI on Δ": f"[{test['diff_lo']:+.4f}, {test['diff_hi']:+.4f}]",
                "McNemar p": f"{test['p']:.2e}" if test["p"] < 0.001 else f"{test['p']:.3f}",
                "verdict": "distinguishable" if test["p"] < 0.05 else "NOT distinguishable",
            }
        )
    print("\n" + pd.DataFrame(rows).to_markdown(index=False))
    return 0


def score_candidates(model, users: np.ndarray, depth: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Top-*depth* ids and scores for every user, timed and announced."""
    t0 = time.perf_counter()
    ids, scores = model.recommend_scored(users, k=depth)
    print(f"  {label}: top-{depth} for {len(users):,} users in {time.perf_counter() - t0:.0f}s", flush=True)
    return ids, scores


def apply_rule(cf: tuple[np.ndarray, np.ndarray], ct: tuple[np.ndarray, np.ndarray], *, k: int, **kwargs) -> np.ndarray:
    """Run one combination rule over precomputed candidate lists for every user."""
    cf_ids, cf_scores = cf
    ct_ids, ct_scores = ct
    out = np.full((len(cf_ids), k), None, dtype=object)
    for row in range(len(cf_ids)):
        picked = combine_user(cf_ids[row], cf_scores[row], ct_ids[row], ct_scores[row], k=k, **kwargs)
        out[row, : len(picked)] = picked
    return out


def metrics(recommended: np.ndarray, bench: Bench, holdout: np.ndarray, k: int) -> dict[str, float]:
    """The same three metrics the comparison table reports, from a recommendation array."""
    popularity = dict(zip(bench.train.item_ids.tolist(), bench.train.item_popularity.tolist(), strict=True))
    total = int(bench.train.item_popularity.sum())
    hits = hit_vector(recommended, holdout)
    filled = sum(1 for row in recommended.ravel().tolist() if row is not None)
    return {
        f"HitRate@{k}": float(hits.mean()),
        f"Coverage@{k}": catalog_coverage_at_k(recommended, bench.catalog_ids, bench.catalog_size),
        f"Novelty@{k}": novelty_at_k(recommended, popularity, total, bench.catalog_size),
        "slots filled": filled / recommended.size,
    }


def attribution(recommended: np.ndarray, base: np.ndarray, holdout: np.ndarray) -> dict[str, float]:
    """The L58-style honesty column: how much of this row is item-item wearing a hat?

    ``shared slots`` is how much of the *list* the collaborative model would have produced
    anyway; ``hits it already had`` is how many of the hybrid's hits item-item also found,
    and ``hits it did not`` is the number the hybrid actually bought.
    """
    hybrid_hits = hit_vector(recommended, holdout)
    base_hits = hit_vector(base, holdout)
    shared = 0
    for row in range(len(recommended)):
        a = {i for i in recommended[row].tolist() if i is not None}
        b = {i for i in base[row].tolist() if i is not None}
        shared += len(a & b)
    filled = sum(1 for i in recommended.ravel().tolist() if i is not None)
    return {
        "shared slots": shared / max(filled, 1),
        "hits it already had": int(np.sum(hybrid_hits & base_hits)),
        "hits it did not": int(np.sum(hybrid_hits & ~base_hits)),
        "hits it lost": int(np.sum(~hybrid_hits & base_hits)),
    }


def loose_title_key(title: str) -> str:
    """A title stripped to what an edition, a format and a subtitle all share.

    Everything from the first ``:`` or ``(`` onwards is dropped, then everything that is
    not alphanumeric. So *The Lovely Bones: A Novel*, *The Lovely Bones* and *Harry Potter
    and the Sorcerer\'s Stone: A Deluxe Pop-up Book* collapse onto the book they are all
    an edition or a companion of.

    **What it catches and what it does not, stated because the number is only readable
    with both.** It catches the subtitle, format and companion classes — the ones L64\'s
    note scopes at 9,523 further works. It does **not** catch translations: *El Codigo Da
    Vinci* shares no characters with *The Da Vinci Code*, and L47 is the standing record
    that title equality is this project\'s ceiling. So every duplicate share below is a
    **lower bound** on what a reader would call "a book I already have".
    """
    head = title.split(":")[0].split("(")[0]
    return "".join(ch for ch in head.lower() if ch.isalnum())


def duplicate_title_share(
    recommended: np.ndarray,
    users: np.ndarray,
    bench: Bench,
    key_of: dict[str, str],
) -> dict[str, float]:
    """Share of slots whose loose title is already in the reader\'s train profile.

    This is ledger L45\'s question asked of a *work-level* list, where the published key
    says there is nothing left to deduplicate. The gallery says otherwise, and this is the
    number behind the gallery: a recommendation the reader would read as "the book I am
    holding" scores as a legitimate work under the M12 key.
    """
    train = bench.train
    filled = duplicate = affected = 0
    for row, user in enumerate(users):
        user_row = train.user_index.get(int(user))
        if user_row is None:
            continue
        lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
        owned = {key_of.get(i, i) for i in train.item_ids[train.matrix.indices[lo:hi]].tolist()}
        slots = [i for i in recommended[row].tolist() if i is not None]
        same = sum(key_of.get(i, i) in owned for i in slots)
        filled += len(slots)
        duplicate += same
        affected += same > 0
    return {
        "dup-title slots": duplicate / filled if filled else float("nan"),
        "dup-title users": affected / len(users),
    }


SUPPORT_STRATA = [(0, 0, "sup 0"), (1, 4, "1-4"), (5, 49, "5-49"), (50, 10**9, "50+")]


def net_by_stratum(recommended: np.ndarray, base_hits: np.ndarray, holdout: np.ndarray, support: np.ndarray) -> dict:
    """Where a rule's net change comes from, split by the held-out work's train support.

    This is the table that says whether a hybrid *adds* or *redistributes*. The coverage
    argument for a hybrid (L50, L73) predicts gains in the leftmost stratum and nowhere
    else; a rule that also loses hits at 50+ is trading well-evidenced readers for thin
    ones, which is a different product decision wearing the same HitRate.
    """
    hits = hit_vector(recommended, holdout)
    won, lost = hits & ~base_hits, ~hits & base_hits
    out: dict = {"net": int(won.sum()) - int(lost.sum())}
    for lo, hi, label in SUPPORT_STRATA:
        mask = (support >= lo) & (support <= hi)
        out[label] = f"+{int(won[mask].sum())}/-{int(lost[mask].sum())}"
    return out


def new_hit_attribution(
    recommended: np.ndarray,
    base: np.ndarray,
    users: np.ndarray,
    holdout: np.ndarray,
    bench: Bench,
    key_of: dict[str, str],
) -> dict[str, int]:
    """Of the hits a rule wins over item-item, how many are the reader\'s own book back?

    A rule can raise HitRate by finding the held-out work legitimately, or by returning
    something whose loose title the reader already owns — and at work level the second is
    invisible to the metric, because those are different work ids under the M12 key. This
    is the number the gallery demanded: *The Lovely Bones* beside *The Lovely Bones: A
    Novel* is a hit the offline table scores and a reader would call a bug.

    Same lower-bound caveat as :func:`duplicate_title_share`: translations do not match.
    """
    won = hit_vector(recommended, holdout) & ~hit_vector(base, holdout)
    train = bench.train
    duplicate = 0
    for row in np.flatnonzero(won).tolist():
        user_row = train.user_index.get(int(users[row]))
        if user_row is None:
            continue
        lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
        owned = {key_of.get(i, i) for i in train.item_ids[train.matrix.indices[lo:hi]].tolist()}
        target = str(holdout[row])
        duplicate += key_of.get(target, target) in owned
    return {"new hits": int(won.sum()), "...of a book already owned": duplicate}


def fit_pair(bench: Bench, *, label: str) -> tuple:
    """Fit the collaborative and the content model on *bench*'s train, unchanged."""
    models = []
    for name in ("item-item", "tfidf"):
        t0 = time.perf_counter()
        model = build_model(name, work_level=True)
        fit_model(model, bench.train, bench.catalog, bench.split.train)
        print(f"  {label} {name} fitted in {time.perf_counter() - t0:.0f}s", flush=True)
        models.append(model)
    return tuple(models)


def check_bases(models, bench: Bench, holdout: np.ndarray, k: int) -> tuple[int, dict[str, np.ndarray]]:
    """Both base rows reproduce to the digit, and bit for bit against M18's vectors.

    Returns the exit code and the two models' **published** top-k arrays. Those arrays are
    the honest baseline for the attribution column: the first k columns of a top-100 call
    are *not* the same list, because ``argpartition`` resolves ties differently at a
    different depth — the artefact ``scripts/analyze_dedup.py`` measured at 0.0546 against
    0.0543. Comparing a hybrid against a sliced list would credit or blame it for that.
    """
    from recommender.eval import evaluate

    published_lists: dict[str, np.ndarray] = {}
    cached = None
    try:
        cached = np.load(HIT_CACHE, allow_pickle=False)
    except FileNotFoundError:
        print(f"  note: {HIT_CACHE} absent — the bit-for-bit check is skipped, four decimals is all this run has")

    failures = []
    for name, model in zip(("item-item", "tfidf"), models, strict=True):
        result = evaluate(
            model, bench.split, bench.train, catalog_isbns=bench.catalog_ids, catalog_size=bench.catalog_size, k=k
        )
        want = PUBLISHED[name]
        got = (
            round(result.hit_rate_at_10, 4),
            round(result.coverage_at_10 * 100, 3),
            round(result.novelty_at_10, 2),
        )
        published_lists[name] = result.recommendations
        if got != want:
            failures.append(f"{name}: measured {got}, published {want}")
        if cached is not None:
            same = np.array_equal(hit_vector(result.recommendations, holdout), cached[name].astype(bool))
            if not same:
                failures.append(f"{name}: per-user hit vector differs from M18's cached run")
        print(f"  {name}: {got} against published {want}{'' if cached is None else ', vectors identical'}", flush=True)

    if failures:
        for line in failures:
            print(f"REPRODUCTION FAILURE  {line}")
        print("\nA base model moved. The hybrid is not the finding here — stop and escalate.")
        return 1, published_lists
    return 0, published_lists


def tune_alpha(outer: Bench, *, k: int, depth: int) -> pd.DataFrame:
    """Sweep alpha on the inner validation split, and report the whole curve."""
    inner = inner_bench(outer, seed=43)
    print(f"\ninner validation split (seed 43): {inner.split.describe()}", flush=True)
    collaborative, content = fit_pair(inner, label="inner")
    users = inner.split.test["User-ID"].to_numpy()
    holdout = inner.split.test["ISBN"].to_numpy()
    cf = score_candidates(collaborative, users, depth, "inner item-item")
    ct = score_candidates(content, users, depth, "inner tfidf")

    rows = []
    for alpha in ALPHAS:
        recommended = apply_rule(cf, ct, k=k, rule=FUSION, alpha=alpha)
        rows.append({"alpha": alpha, **metrics(recommended, inner, holdout, k)})
        print(f"  alpha={alpha:.1f}  HitRate@{k}={rows[-1][f'HitRate@{k}']:.4f}", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--alpha", type=float, default=None, help="skip tuning and use this alpha for the fusion row")
    parser.add_argument("--tune", action="store_true", help="sweep alpha on the inner validation split first")
    parser.add_argument("--gallery", action="store_true", help="print the three-anchor gallery for every rule")
    parser.add_argument("--floors", type=int, nargs="*", default=[0, 5, 20], help="cascade support floors to measure")
    parser.add_argument("--save", default=None, help="write every rule\'s top-k array to this .npz")
    parser.add_argument(
        "--pairs-from",
        default=None,
        help="paired McNemar between every pair of rules in a saved .npz, and nothing else (no re-fit)",
    )
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="smoke-test guardrail: score a seeded sample. Rows from a sampled run are NOT ledger rows.",
    )
    args = parser.parse_args(argv)

    if args.pairs_from:
        return pairwise_report(args.pairs_from, HIT_CACHE)

    started = time.perf_counter()
    bench = build_bench(work_level=True)
    print(bench.describe())
    holdout = bench.split.test["ISBN"].to_numpy()
    users = bench.split.test["User-ID"].to_numpy()
    if args.limit_users:
        rng = np.random.default_rng(bench.split.seed)
        keep = np.sort(rng.choice(len(users), size=min(args.limit_users, len(users)), replace=False))
        users, holdout = users[keep], holdout[keep]
        print(f"SAMPLED RUN: {len(users):,} users. Nothing from this run may become a ledger row.")
    print(f"data ready in {time.perf_counter() - started:.0f}s\n", flush=True)

    print("fitting the two base models — unchanged, and checked before anything is combined")
    collaborative, content = fit_pair(bench, label="outer")
    published_lists: dict[str, np.ndarray] = {}
    if args.limit_users:
        print("  reproduction check skipped: it is only meaningful on the full split")
    else:
        code, published_lists = check_bases((collaborative, content), bench, holdout, args.k)
        if code:
            return 1

    alpha = args.alpha
    curve = None
    if args.tune:
        curve = tune_alpha(bench, k=args.k, depth=args.candidates)
        best = curve.loc[curve[f"HitRate@{args.k}"].idxmax()]
        alpha = float(best["alpha"])
        print(f"\nalpha curve on inner validation:\n{curve.round(5).to_markdown(index=False)}")
        print(f"\nchosen on validation: alpha={alpha:.1f}. The curve above is the result; the argmax is a summary.")
    if alpha is None:
        alpha = 0.5
        print("\nno tuning requested — the fusion row uses alpha=0.5 and says so")

    print("\nscoring the candidate lists once, for all three rules", flush=True)
    cf = score_candidates(collaborative, users, args.candidates, "item-item")
    ct = score_candidates(content, users, args.candidates, "tfidf")

    # The baseline for attribution is item-item's *published* list where we have it, and the
    # sliced one only in a sampled smoke run — with the difference between the two reported,
    # because it is the tie artefact and not a rounding detail.
    sliced = cf[0][:, : args.k]
    base_top_k = published_lists.get("item-item", sliced)
    if published_lists:

        def filled(row: np.ndarray) -> list[str]:
            return [i for i in row.tolist() if i is not None]

        differing = sum(1 for row in range(len(sliced)) if filled(sliced[row]) != filled(base_top_k[row]))
        print(
            f"  tie artefact: the first {args.k} of item-item's top-{args.candidates} differ from its "
            f"published top-{args.k} for {differing:,} of {len(sliced):,} users "
            f"({differing / len(sliced):.2%}); attribution uses the published list"
        )
    rows, galleries, saved, strata_rows = [], {}, {}, []
    holdout_support = np.array(
        [
            bench.train.item_popularity[bench.train.item_index[i]] if i in bench.train.item_index else 0
            for i in holdout.tolist()
        ]
    )
    variants = [(f"cascade (floor {floor})", {"rule": CASCADE, "support_floor": floor}) for floor in args.floors]
    variants += [(f"fusion (alpha {alpha:.1f})", {"rule": FUSION, "alpha": alpha}), ("rrf (k=60)", {"rule": RRF})]

    support = dict(zip(bench.train.item_ids.tolist(), bench.train.item_popularity.tolist(), strict=True))
    key_of = {
        str(isbn): loose_title_key(str(title))
        for isbn, title in zip(bench.catalog.books["ISBN"], bench.catalog.books["Book-Title"], strict=True)
    }
    base_hits = hit_vector(base_top_k, holdout)

    # The control row. It is not a hybrid: it is item-item's own top-100 cut to k, which is
    # what every rule below actually starts from. Any difference between it and the
    # published row is the tie artefact, and reading a hybrid against the wrong one of the
    # two would credit the rule for a depth change.
    variants = [("item-item, top-100 cut to k (control)", {"rule": CASCADE, "content_off": True})] + variants

    for label, kwargs in variants:
        t0 = time.perf_counter()
        if kwargs.pop("content_off", False):
            recommended = cf[0][:, : args.k].copy()
        else:
            recommended = apply_rule(cf, ct, k=args.k, support=support, **kwargs)
        test = mcnemar(hit_vector(recommended, holdout), base_hits)
        duplicates = duplicate_title_share(recommended, users, bench, key_of)
        won = new_hit_attribution(recommended, base_top_k, users, holdout, bench, key_of)
        strata_rows.append({"rule": label, **net_by_stratum(recommended, base_hits, holdout, holdout_support)})
        rows.append(
            {
                "rule": label,
                **{key: round(value, 5) for key, value in metrics(recommended, bench, holdout, args.k).items()},
                **attribution(recommended, base_top_k, holdout),
                **{key: round(value, 4) for key, value in duplicates.items()},
                **won,
                "McNemar p vs item-item": f"{test['p']:.2e}" if test["p"] < 0.001 else f"{test['p']:.3f}",
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(f"  {label}: {rows[-1]}", flush=True)
        galleries[label] = kwargs
        saved[label] = recommended.astype(str)

    if args.save:
        import pathlib

        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, holdout=holdout, users=users, **saved)
        print(f"\nrecommendation arrays written to {args.save}")

    print("\n" + pd.DataFrame(rows).to_markdown(index=False))
    print("\nWhere the net change comes from, by the held-out work's train support:")
    print(pd.DataFrame(strata_rows).to_markdown(index=False))
    print(
        "\nA rule that gains only in the leftmost stratum is the coverage argument working. "
        "One that also loses at 50+ is redistributing accuracy from well-evidenced readers to "
        "thin ones — the same HitRate, a different product decision."
    )
    print(
        "\nRead 'hits it did not' against 'hits it lost': the first is what the content layer "
        "bought, the second is what displacing a collaborative slot cost. A rule is only worth "
        "shipping if the first exceeds the second, and 'shared slots' says how much of the list "
        "item-item would have produced on its own."
    )

    if args.gallery:
        for label, kwargs in galleries.items():
            model = HybridRecommender(collaborative, content, candidates=args.candidates, name=label, **kwargs)
            print(f"\n### {label}\n")
            print(render_markdown(build_gallery([model], bench.catalog, anchors=bench.anchors, k=args.k)))

    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
