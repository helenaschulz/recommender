"""Ten slots is a UI choice, not a claim. What would a score-gap rule cut? (M14.5)

    python scripts/analyze_truncation.py
    python scripts/analyze_truncation.py --taus 0.5 0.6 0.7 --anchors 300

**This script proposes; it does not wire anything in.** M14.5 asks for a rule derived from
the data, the number of slots it removes across the eleven demo anchors, and Helena's
verdict *before* it reaches ``demo.similar``.

The observation behind it: *Harry Potter*, *The Hobbit* and *Bridget Jones* all have a
strong head and a noise tail. Once the same-author or same-series cluster is exhausted
there is not enough shared readership left to fill ten slots, and the list ends in *Fever
1793*, *Magic the Gathering: Arena* and *Lizard*.

**The rule family, and why this one.** Keep slot *i* while ``score_i >= tau * score_1`` —
a *relative* drop against the anchor's own top score. It has to be relative, and that is
exactly what ledger L63 proves: absolute cosines are not comparable across anchors, so an
absolute cutoff would empty a well-supported anchor's list and keep a thin one's intact.
Two alternatives are measured beside it so the choice is not a single option presented as
inevitable: the largest *consecutive* gap (an elbow), and the same relative rule with a
minimum list length.

**What this rule cannot fix, and it is the important half.** A relative rule is blind to
whether the head itself is trustworthy: *Guns, Germs, and Steel* has 67 readers, its top
score is 0.507 and its tenth is 0.383, so nothing is cut — every slot is within 76% of the
top one, and every slot rests on 1-6 shared readers. The anchor-side support floor (M14.2)
is the lever for that anchor; truncation is the lever for a *good* anchor's tail. They are
different problems and the deck should not conflate them.

M14.5 rules out one candidate up front — a threshold on the **raw** co-reader count —
because *Fight Club* -> *Trainspotting* (7 co-readers) and *A Clockwork Orange* (4) are
both excellent and any raw threshold that removes noise removes them too. That objection
does not survive **normalising by the anchor's own readership**, and this script measures
that third rule beside the other two, because the data says it is the better one: seven
shared readers out of *Fight Club*'s 102 is 6.9% of the anchor's audience, while *Lizard*'s
twelve out of *Bridget Jones's Diary*'s 772 is 1.6%. Raw counts put *Lizard* ahead; the
share puts it where a reader would. The rule is ``co_readers / anchor_readers >= rho``, and
it has the same scale-freeness that makes the relative score rule preferable to an absolute
cutoff — for the same underlying reason (L63).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from recommender.data import split_series
from recommender.demo import DemoEngine, Suggestion, load_assets
from recommender.gallery import DEMO_ANCHORS

DEFAULT_TAUS = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
DEFAULT_RHOS = [0.01, 0.02, 0.03, 0.05]


def keep_relative(suggestions: list[Suggestion], tau: float, minimum: int = 1) -> int:
    """How many leading slots survive ``score_i >= tau * score_1``, never fewer than *minimum*."""
    if not suggestions:
        return 0
    cutoff = suggestions[0].evidence.score * tau
    kept = sum(1 for s in suggestions if s.evidence.score >= cutoff)
    return max(kept, min(minimum, len(suggestions)))


def keep_share(suggestions: list[Suggestion], rho: float, minimum: int = 1) -> int:
    """How many leading slots reach ``co_readers / anchor_readers >= rho``.

    Leading, not "how many pass": the output is a truncation, so the first failing slot
    ends the list even if a later one would have passed. That keeps the rule a *tail* rule
    rather than a filter that reorders what the model ranked.
    """
    if not suggestions:
        return 0
    kept = 0
    for item in suggestions:
        readers = item.evidence.anchor_readers or 1
        if item.evidence.co_readers / readers < rho:
            break
        kept += 1
    return max(kept, min(minimum, len(suggestions)))


def keep_elbow(suggestions: list[Suggestion]) -> int:
    """Cut at the largest drop between consecutive scores."""
    if len(suggestions) < 2:
        return len(suggestions)
    scores = [s.evidence.score for s in suggestions]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    return int(np.argmax(gaps)) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taus", type=float, nargs="*", default=DEFAULT_TAUS)
    parser.add_argument("--rhos", type=float, nargs="*", default=DEFAULT_RHOS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=300, help="random anchors for the aggregate view")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    loaded = load_assets()
    assets = replace(loaded, factors=np.array(loaded.factors))
    engine = DemoEngine(assets)

    # -- 1 · The eleven anchors, slot by slot ----------------------------------------
    answers = {label: engine.similar(work, k=args.k, tau=0) for work, label in DEMO_ANCHORS.items()}
    print("=" * 112)
    print("1 · The eleven demo anchors: how many of the ten slots survive")
    print("=" * 112)
    print(
        f"{'anchor':<40} {'readers':>8} {'elbow':>6} "
        + " ".join(f"{'t' + format(tau, '.2f'):>6}" for tau in args.taus)
        + "  "
        + " ".join(f"{'r' + format(rho, '.2f'):>6}" for rho in args.rhos)
    )
    for work, label in DEMO_ANCHORS.items():
        suggestions = answers[label]
        readers = engine.describe(work).readers
        by_tau = " ".join(f"{keep_relative(suggestions, tau):>6}" for tau in args.taus)
        by_rho = " ".join(f"{keep_share(suggestions, rho):>6}" for rho in args.rhos)
        print(f"{label:<40} {readers:>8,} {keep_elbow(suggestions):>6} {by_tau}  {by_rho}")
    slots = sum(len(v) for v in answers.values())
    totals = [sum(keep_relative(v, tau) for v in answers.values()) for tau in args.taus]
    rho_totals = [sum(keep_share(v, rho) for v in answers.values()) for rho in args.rhos]
    elbow_total = sum(keep_elbow(v) for v in answers.values())
    print(
        f"\n{'total slots kept of ' + str(slots):<40} {'':>8} {elbow_total:>6} "
        + " ".join(f"{t:>6}" for t in totals)
        + "  "
        + " ".join(f"{t:>6}" for t in rho_totals)
    )
    print(
        f"{'removed':<40} {'':>8} {slots - elbow_total:>6} "
        + " ".join(f"{slots - t:>6}" for t in totals)
        + "  "
        + " ".join(f"{slots - t:>6}" for t in rho_totals)
    )

    # -- 2 · What exactly each tau removes, so it can be judged ----------------------
    print("\n" + "=" * 112)
    print("2 · The slots each rule removes, by name — the only way to see whether a cut is right")
    print("=" * 112)
    rules: list[tuple[str, object]] = [(f"relative score, tau = {tau:.2f}", tau) for tau in args.taus]
    rules += [(f"co-reader share, rho = {rho:.2f}", rho) for rho in args.rhos]
    for name, value in rules:
        keep = keep_relative if name.startswith("relative") else keep_share
        print(f"\n--- {name} " + "-" * max(4, 90 - len(name)))
        for label in DEMO_ANCHORS.values():
            suggestions = answers[label]
            kept = keep(suggestions, value)
            if kept >= len(suggestions):
                print(f"  {label:<40} keeps all {len(suggestions)}")
                continue
            dropped = ", ".join(
                f"{split_series(s.title)[0][:34]} ({s.evidence.co_readers}"
                f"/{s.evidence.anchor_readers} = {s.evidence.co_readers / max(s.evidence.anchor_readers, 1):.1%})"
                for s in suggestions[kept:]
            )
            print(f"  {label:<40} keeps {kept}, drops: {dropped}")

    # -- 3 · The same rules over a random sample, for the aggregate cost -------------
    support = assets.item_support
    named = set(assets.books.index)
    nameable = np.array([isbn in named for isbn in assets.item_ids.tolist()])
    answerable = np.flatnonzero(nameable & (support >= assets.anchor_floor))
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(answerable, size=min(args.anchors, answerable.size), replace=False)
    sample = [engine.similar(str(assets.item_ids[row]), k=args.k, tau=0) for row in rows.tolist()]
    sample = [s for s in sample if s]

    print("\n" + "=" * 112)
    print(f"3 · Over {len(sample):,} random answerable anchors (seed {args.seed})")
    print("=" * 112)
    total = sum(len(s) for s in sample)
    print(f"{'rule':<22} {'slots kept':>12} {'removed':>10} {'median list':>13} {'lists cut to <5':>17}")
    elbow_kept = [keep_elbow(s) for s in sample]
    print(
        f"{'largest gap (elbow)':<22} {sum(elbow_kept):>12,} {1 - sum(elbow_kept) / total:>9.1%} "
        f"{np.median(elbow_kept):>13.0f} {np.mean([k < 5 for k in elbow_kept]):>16.1%}"
    )
    for tau in args.taus:
        kept = [keep_relative(s, tau) for s in sample]
        print(
            f"{'relative tau=' + format(tau, '.2f'):<22} {sum(kept):>12,} {1 - sum(kept) / total:>9.1%} "
            f"{np.median(kept):>13.0f} {np.mean([k < 5 for k in kept]):>16.1%}"
        )
    for rho in args.rhos:
        kept = [keep_share(s, rho) for s in sample]
        print(
            f"{'co-reader rho=' + format(rho, '.2f'):<22} {sum(kept):>12,} {1 - sum(kept) / total:>9.1%} "
            f"{np.median(kept):>13.0f} {np.mean([k < 5 for k in kept]):>16.1%}"
        )
    # -- 4 · Why a co-reader rule cannot be a *truncation* ---------------------------
    # It looked like the better rule on paper — normalising by anchor readership does
    # separate Trainspotting (6.9% of Fight Club's audience) from Lizard (1.6% of Bridget
    # Jones's). It fails anyway, and section 2 shows why: it drops *The Secret Life of
    # Bees* at 14.3% under The Lovely Bones, because slot 2 failed first. The model's
    # score order is not monotone in the evidence, so no leading cut on evidence is safe.
    correlations = []
    inversions = []
    for suggestions in sample:
        if len(suggestions) < 3:
            continue
        shares = np.array(
            [s.evidence.co_readers / max(s.evidence.anchor_readers, 1) for s in suggestions], dtype=float
        )
        ranks_by_score = np.arange(len(shares), dtype=float)
        order = np.argsort(np.argsort(-shares, kind="stable"), kind="stable").astype(float)
        if shares.std() > 0:
            correlations.append(float(np.corrcoef(ranks_by_score, order)[0, 1]))
        # a "recovery": a slot whose share beats an earlier slot's by 2x or more
        best_so_far = np.maximum.accumulate(shares)
        inversions.append(bool(np.any(shares[1:] >= 2 * best_so_far[:-1])))
    print("\n" + "=" * 112)
    print("4 · Why the co-reader rule cannot be a truncation: the score does not order the evidence")
    print("=" * 112)
    print(
        f"mean Spearman between score rank and co-reader-share rank within a top-10: "
        f"{np.mean(correlations):.2f} (n={len(correlations):,})"
    )
    print(
        f"lists where some slot has at least twice the co-reader share of every slot above it: "
        f"{np.mean(inversions):.1%} of {len(inversions):,}"
    )
    print(
        "\nSo a leading cut on evidence throws away the recoveries, and a *filter* on evidence would\n"
        "reorder what the model ranked — a different product change from the one M14.5 asks for.\n"
        "The relative-score rule is the only one of the three that is a truncation at all."
    )
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
