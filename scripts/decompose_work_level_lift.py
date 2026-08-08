"""Split the work-level re-base lift into evaluation fairness and merged signal.

    python scripts/decompose_work_level_lift.py            # all six models
    python scripts/decompose_work_level_lift.py tfidf      # the row that motivated it

**The question.** Moving the primary table to work level lifted TF-IDF's HitRate@10 by
+77.6%, far outside the plausibility band M12.6 set. No bug was found, and the branch's
gate note argued the band had been calibrated on the wrong prior. That argument is a story
until the lift is taken apart, which is what this script does.

**The two components.** Re-basing does two separable things to a text model:

1. *Evaluation fairness.* At ISBN level a recommendation of the Penguin edition when the
   reader's held-out book was the Vintage edition scores **zero** — the model named the
   right book and the metric called it wrong. Work-level credit fixes only that.
2. *Merged signal.* Co-occurrence counts split across editions become one count, the
   canonical per-work text replaces however many near-duplicate strings, the item universe
   shrinks, and a slot spent on a second edition of the anchor is spent on a real
   candidate instead.

Component 1 is isolated exactly, because **nothing** about the run changes except the
scoring rule: the same ISBN-level model, fitted on the same matrix, producing the same
top-10 lists, re-scored with :func:`recommender.eval.hit_rate_at_k_by_group` so a slot
counts when it is any edition of the held-out work. Component 2 is the residual up to the
full work-level row.

**The hole in component 1, measured rather than argued.** Work credit can hand out a hit
the work-level table could never award: if the reader held out edition A of a work and
still owns edition B of it in train, then recommending edition C scores a hit — for a book
they demonstrably already have. At work level that is impossible, because an owned work is
blocked from the candidate list. So the script reports a **third** column, "work credit,
owned works blocked", which drops any slot whose work is already in the reader's train
profile. It is the conservative reading of component 1, and the gap between the two columns
is exactly how much of the fairness gain is that artefact.

**Why this re-runs the work-level side instead of quoting it.** The work-level column here
is produced by the same code path as the primary comparison table and must agree with it to
the digit. That agreement is the check — the same discipline ``scripts/analyze_dedup.py``
uses when it reproduces every "before" row rather than citing it.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from recommender.benchmark import build_bench
from recommender.data import load
from recommender.eval import evaluate, hit_rate_at_k_by_group
from recommender.models import ALL_MODELS, build_model, fit_model
from recommender.serving import blank_owned_works, duplicate_slot_rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=[])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    names = args.models or ALL_MODELS

    started = time.perf_counter()
    catalog = load()
    works = catalog.works
    isbn = build_bench(work_level=False, catalog=catalog)
    work = build_bench(work_level=True, catalog=catalog)
    print(f"ISBN level: {isbn.describe()}")
    print(f"work level: {work.describe()}")
    print(f"data ready in {time.perf_counter() - started:.0f}s\n", flush=True)

    rows = []
    for name in names:
        isbn_model = fit_model(build_model(name), isbn.train, isbn.catalog, isbn.split.train)
        isbn_result = evaluate(
            isbn_model, isbn.split, isbn.train, catalog_isbns=isbn.catalog_ids, catalog_size=isbn.catalog_size, k=args.k
        )
        holdout = isbn.split.test["ISBN"].to_numpy()
        users = isbn.split.test["User-ID"].to_numpy()
        # The only thing that differs between these two numbers is the definition of a hit.
        fair = hit_rate_at_k_by_group(isbn_result.recommendations, holdout, works.of)
        # ...and the conservative version, which refuses credit for a work already owned.
        strict_fair = hit_rate_at_k_by_group(
            blank_owned_works(isbn_result.recommendations, users, isbn.train, works), holdout, works.of
        )
        # Ledger L45's column, recomputed here because it is what the lift correlates with.
        dup = duplicate_slot_rate(isbn_result.recommendations, users, isbn.train, works)

        work_model = fit_model(build_model(name, work_level=True), work.train, work.catalog, work.split.train)
        work_result = evaluate(
            work_model, work.split, work.train, catalog_isbns=work.catalog_ids, catalog_size=work.catalog_size, k=args.k
        )

        base, full = isbn_result.hit_rate_at_10, work_result.hit_rate_at_10
        rows.append(
            {
                "model": name,
                "ISBN, ISBN credit": round(base, 4),
                "ISBN, work credit": round(fair, 4),
                "...owned works blocked": round(strict_fair, 4),
                "work level": round(full, 4),
                "evaluation fairness": f"{fair / base - 1:+.1%}",
                "merged signal": f"{full / fair - 1:+.1%}",
                "total": f"{full / base - 1:+.1%}",
                "ISBN dup slots": f"{dup['duplicate_slot_share']:.1%}",
            }
        )
        print(
            f"{name:<20} {base:.4f} -> {fair:.4f} (fairness, {strict_fair:.4f} owned-blocked)"
            f" -> {full:.4f} (full)",
            flush=True,
        )

    print("\n" + pd.DataFrame(rows).to_markdown(index=False))
    print(
        "\nRead the middle columns as one sentence: the ISBN-keyed metric was scoring "
        "right answers as wrong, and it was doing it hardest to the models whose output "
        "duplicated editions most (ledger L45). The 'owned works blocked' column is the "
        "same measurement refusing credit for a work the reader already had — the gap "
        "between it and 'work credit' is the part of the fairness gain that work-level "
        "scoring could never have awarded."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
