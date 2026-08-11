# Change log

What was built, in the order it was built, with the ledger lines each step produced. Every
number below is a pointer, not a source: [`RESULTS.md`](RESULTS.md) is the authority, and
where the two disagree the ledger wins. For the argument rather than the sequence, see
[`SUMMARY.md`](SUMMARY.md).

Milestones are numbered M1…M23.10 in the order they were planned, which is also the order
they were run.

---

## 2026-08-03 · Getting the data honestly described

**M1–M2 · Repository.** Package skeleton on `main`, `.gitignore` verified *before* the first
`git add`, so no raw CSV and no working note ever entered a commit.

**M3 · The EDA notebook.** [`01_eda.ipynb`](../notebooks/01_eda.ipynb) executed top to
bottom on a fresh kernel; its section 9 re-verifies every claim in
[`dataset_findings.md`](dataset_findings.md) and printed **27/27 match**. The three
properties that decided everything after it: 62.3% of interactions carry no grade, density
is 0.0032%, and collaborative filtering can reach 5.3% of the catalogue.
→ **L1–L18**

## 2026-08-04 · The harness, then the ladder

**M4 · Split and evaluation harness.** `src/recommender/` (data, split, eval, models, gallery),
`scripts/run_model.py`, 25 tests. The split is pinned once here and never re-drawn for a
result: per-user leave-one-out, seed 42, relevance ≥8. The structural ceilings are measured
before any model, so every later score can be read against what was reachable at all.
→ **L19–L21**

**M5 · Popularity baseline.** 0.0145 / **0.019%** / 10.81. That coverage figure is the point
of running it: the baseline reaches two hundredths of one percent of the catalogue, which is
why coverage and novelty are reported beside accuracy from here on.
→ **L22**

**M6 · Item-item CF, the core hypothesis.** Shrunk cosine, λ=10, 50 neighbours: **0.0546**
and 9.064% coverage, against the baseline's 0.0145 and 0.019%. The explicit-only ablation
scores 0.0379, so discarding the ungraded rows costs 31% of the hit rate.
→ **L22, L24, L26**

**M7 · Content TF-IDF.** Character 3–5-grams over title and author, all 271,360 books:
0.0228 / **16.616%** — the widest reach of anything measured so far, at the lowest accuracy
of the three real models. The two classes turned out to be nearly disjoint in *what* they
reach, which is the measurement the hybrid argument rests on.
→ **L30**, later **L60**

**M8 · ALS / weighted MF.** 128 factors: 0.0451 / 0.835%. It loses to item-item on all three
metrics and is the most popularity-biased model in the table — recorded as a negative result
rather than tuned until it won.
→ **L33**

**M9 · Multilingual embeddings.** 0.0109 / **23.911%** / novelty **18.42**. The naive recipe
scored 0.0036 on validation until item vectors were centered; averaging uncentered sentence
embeddings gives almost every user the same profile (mean cosine 0.883 to the global
centroid).
→ **L35, L36**

**M10 · Synthesis.** The six-model comparison table, [`02_models.ipynb`](../notebooks/02_models.ipynb)
executed fresh-kernel against the package with nothing re-implemented, and the first written
account of the choice.

## 2026-08-08 · The data turns out to matter more than the models

**M11 · Edition clustering, measured before deciding.** The catalogue is **235,824 works
behind 271,360 ISBNs**; the earlier estimate had undercounted the duplication by 47%.
Clustering *before* training is worth **+18%** on item-item — the largest single accuracy
gain in the project, and it is data preparation, not modelling. 30 clusters read by hand,
**0 wrong merges**; the one-character surname extension was audited separately and its single
error is reported rather than tuned away.
→ **L40–L48**

**M12 · The whole table re-bases to works.** Every model moved, and not by the same amount.
TF-IDF moved **+77.4%**, outside the plausibility band the milestone had set for itself in
advance — the gate fired, the branch was held unmerged, and the lift was decomposed into
*evaluation fairness* (+5.4% to +11.4%, roughly uniform) and *merged signal* (+0.5% to
+62.3%, tracking each model's duplicate-slot rate) before anything was published.
→ **L49–L60**

**M13 · The demo.** `streamlit run app/main.py`: paste a book, get ten with a grounded reason
each, no language model in the hot path. The free-text lookup needed two serving rules and
both are audited rather than asserted.
→ **L61–L62**

**M14 · Reading the demo's output instead of a table.** Eleven anchors read by hand found what
no cell shows: **the similarity score is not comparable across anchors**. The floor therefore
became two numbers — anchor floor 50, candidate floor 20. A truncation rule was measured,
wired in, and reverted on seeing it run: a list that ends at four reads as a broken app.
→ **L63–L69**

**M15 · The demo's surface.** Display only, by construction, so no published number could
move — and none did.

## 2026-08-09 · Making the numbers survive being checked

**M17 · Say what the list is sorted by.** The list was sorted by a quantity that was nowhere
on screen, while the only visible quantity contradicted the order. Both are now shown. The
evidence divider was removed: it drew a contiguous line from a criterion that is not monotone
in the sort order, which is this project's own headline finding.
→ **L70**

**M18 · The ledger audit.** **479 numeric literals** across every artefact a reader can open,
each checked against the ledger: **six were wrong**, all of them numbers that were correct
when written and were overtaken by a later measurement. Uncertainty was quantified for the
first time — 95% Wilson intervals (**±0.002 to ±0.004**) and paired McNemar, which declines
to order exactly one pair: embeddings against the baseline, p = 0.342. The serving footprint
was measured for the productionization argument: 890 MB shipped, of which the recommender is
17.5%, against a 1.5 MB answer table for the whole product.
→ **L71–L75**

**M19 · The hybrid, measured at last.** Argued from bounds for four milestones, then run with
the prediction written down beforehand. The recommended cascade is worth **nine readers out
of 13,580** with nobody worse off. The two higher-scoring rules were rejected on a hand read:
they fill 28.8% and 33.9% of readers' lists with another edition of a book they already own.
→ **L76–L79**

## 2026-08-10 · The question the demo actually asks

**M20 · The item query gets its own metric.** AnchorHitRate@10 over 13,580 anchors: ALS
0.0308, item-item 0.0297, and the two **cannot be told apart** (p = 0.47). The same run
settled something larger — ALS scores 0.0308 with the support floor and **0.0130 without it**,
below the popularity baseline, at p = 2.5e-41. **The floor is worth more than the model.**
→ **L80–L82**

**M21 · Every number is conditional on one draw.** Five independent draws: the ordering holds
on all five, no row changes place, and three near-ties change their significance verdict
between draws. A gate run before any fitting found that the embedding cache **cannot detect
the drift it exists to detect** — it samples 0.22% of the catalogue and missed a real change
on four seeds out of four. Reused knowingly, with the error bounded and printed.
→ **L84, L86**

**M22 · The fusion rules on the item query.** RRF 0.0371 and score fusion 0.0355 beat the
cascade on the aggregate — and go **41/42, p = 1.000** against item-item in the band the app
actually serves. Every point of the win is bought below the anchor floor.
→ **L85**

**M23 · The floor and the engine are one decision.** Tier 1 delivered three selectable
configurations; **Tier 2 was stopped by its own gate**. 240 slots read by hand in the band a
lower floor would open killed the text-blended configuration: 17 of its 80 slots were
text-match artefacts, answering an author's name with a different author of the same first
name at zero shared readers.
→ **L87–L89**

**M23.10 · Two engines, one floor.** ALS (default) and item-item ship behind a picker at the
same anchor floor, so the click moves one variable. **440 slots read by hand: 7 bad of 220
for A, 0 of 220 for B.** The switch costs nothing measurable — a 0.22 MB answer table built
in 21.8 s, cold start 8.8 s / 8.5 s, and the rehearsed engine byte-identical on all 110
rehearsed slots. One constant reverts to a single engine.
→ **L90–L91**
