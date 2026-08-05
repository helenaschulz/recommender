# RESULTS.md — the measurement ledger

Every number that appears anywhere in this project — a README claim, a notebook takeaway,
a statement like "beats the baseline by X" — traces to a line in this file. **New claim,
new line**, including the negative results. If a number is not here, it does not get
quoted.

Each line records *how* it was measured, not just what came out, so any number can be
re-derived or challenged. `L1.` … numbering is stable; everything else cites line IDs.

Column meanings: **Source** is the artefact that produced the number (notebook section,
script, or run). **Measured** is the date it was last recomputed.

---

## Data understanding (Book-Crossing, raw CSVs in `data/`)

Source for all lines below: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb),
executed top to bottom on a fresh kernel. Section 9 of that notebook re-verifies every
line against [`dataset_findings.md`](dataset_findings.md) and printed **27/27 match**.

| ID | Claim | Number | How measured | Source | Measured |
|---|---|---|---|---|---|
| L1 | Dataset size | 1,149,780 ratings · 278,858 users · 271,360 books | Row counts of the three CSVs after load | §1 | 2026-08-03 |
| L2 | `Books.csv` contains parser-breaking records | 3 rows repaired, 0 dropped | Rows whose `Year-Of-Publication` is non-numeric — the fingerprint of an unescaped `\";` merging title+author and shifting all later columns by one; repaired, not skipped | §1 | 2026-08-03 |
| L3 | Publication year is unusable as-is | 4,618 rows with year 0; 23 rows after 2006 | `pd.to_numeric(errors="coerce")` on `Year-Of-Publication`; the crawl is from 2004, so >2006 is impossible | §1 | 2026-08-03 |
| L4 | **Most of the data carries no grade** | 716,109 of 1,149,780 = **62.3%** implicit | Count of `Book-Rating == 0`; in Book-Crossing a 0 marks an interaction, not a score | §2 | 2026-08-03 |
| L5 | Explicit ratings are few and left-skewed | 433,671 (37.7%), mean 7.60, mode 8 | `Book-Rating > 0`; mean/mode over that subset | §2 | 2026-08-03 |
| L6 | The interaction matrix is extremely sparse | **0.0032%** density — 1 filled cell in ~31,184 | `n_ratings / (distinct users × distinct ISBNs in Ratings.csv)` | §3 | 2026-08-03 |
| L7 | Most books are rated exactly once | **57.9%** (87.1% have fewer than 5) | Share of `groupby("ISBN").size()` equal to 1 (resp. < 5) | §3 | 2026-08-03 |
| L8 | Most users rate exactly once | 56.2% | Share of `groupby("User-ID").size()` equal to 1 | §3 | 2026-08-03 |
| L9 | **Interactions concentrate in the head** | top 1% of books (3,406 titles) = **25.1%** of all interactions | Sum of the 1% highest per-book counts / total ratings | §3 | 2026-08-03 |
| L10 | The standard filter funnel | 1,149,780 → 433,671 → **152,280** ratings (13.2% survive) | Explicit-only, then ≥5 ratings per user **and** per book, both thresholds evaluated on the explicit set in a single pass | §4 | 2026-08-03 |
| L11 | What survives the funnel | 13,305 users · 14,513 books · density 0.079% (25× denser) | Distinct keys in the filtered set (L10) | §4 | 2026-08-03 |
| L12 | **Pure CF can only reach a sliver of the catalogue** | **5.3%** (14,513 of 271,360 books) | L11 book count / `len(Books.csv)`. This is the quantified argument for a content-based layer — a coverage argument, not a cold-start footnote | §4 | 2026-08-03 |
| L13 | *Negative result:* the standard filter does not keep its own promise | Iterated to a fixed point: 118,668 ratings · 7,025 users · 9,432 books — **22% smaller** than L10 | Re-applying the min-5 filter until stable (10 passes). Removing sparse users pushes books back below the threshold. Reported numbers use the single pass because that is what the public notebooks we compare against do — stated, not hidden | §4 | 2026-08-03 |
| L14 | Ratings that cannot be joined to a book | 118,644 = **10.3%** (70,405 distinct ISBNs) | `~ratings["ISBN"].isin(set(books["ISBN"]))`. Usable for CF, unusable for display | §5 | 2026-08-03 |
| L15 | The same work is split across editions | 17,554 works over 40,675 ISBNs (15.0% of the catalogue) | `groupby` on lower-cased, whitespace-stripped title+author with more than one distinct ISBN. On raw strings it is 15,746 / 35,921 — the normalization matters and is part of the claim | §5 | 2026-08-03 |
| L16 | Most users in `Users.csv` never rated anything | 173,575 = 62.2% | User-IDs in `Users.csv` absent from `Ratings.csv` | §6 | 2026-08-03 |
| L17 | The demographic columns do not carry weight | Age 39.7% missing; range 0–244; 0.74% implausible | Missing share over all users; implausible = age <5 or >100 **as a share of the ages that exist** (0.45% if taken over all users — the base matters) | §6 | 2026-08-03 |
| L18 | **The dataset has no time dimension** | 0 timestamp columns | `Ratings.csv` has exactly `User-ID`, `ISBN`, `Book-Rating`. A temporal split is therefore impossible; evaluation uses per-user leave-N-out, and we say so explicitly rather than describing a split this data cannot support | §7 | 2026-08-03 |

## The evaluation split and its ceilings

Source: `src/recommender/split.py`, verified by `tests/test_split.py`. Every model row
below cites **L19**; without it, no two model numbers are comparable.

| ID | Claim | Number | How measured | Source | Measured |
|---|---|---|---|---|---|
| L19 | **The split, pinned once** | **13,581 eligible users**, 1,136,199 train interactions, train matrix 105,283 × 338,496 | Per-user leave-one-out, **seed 42**. Eligible = ≥5 explicit ratings **and** ≥1 explicit rating ≥8. For each eligible user exactly one item is held out, drawn seeded-at-random from their ratings ≥8; everything else — including all 716,109 implicit interactions and every interaction of non-eligible users — is train. Eligible users are 12.9% of the 105,283 users who rated anything. Fit, similarities, popularity and IDF statistics all come from train only | `split.py` | 2026-08-04 |
| L20 | **A collaborative model cannot exceed 84.8% HitRate here — by construction** | ceiling **84.81%** (11,518 of 13,581 held-out items) | Share of held-out items that appear at all in the train matrix. The other 15.2% were that book's only interaction, so no co-occurrence model can rank an item it has never seen. This is a property of the data, not of any model, and it is the honest denominator to read every HitRate against | `split.py` + train matrix | 2026-08-04 |
| L21 | **The content layer raises that ceiling, and a hybrid raises it further** | content **89.31%** · union of both **95.37%** | Share of held-out items reachable by a content model (has a row in `Books.csv`, so it can be embedded even with zero interactions) and by either model class. The 10.6-point gap between L20 and the union is the coverage argument for the hybrid stated as a bound on achievable accuracy, not as a slogan | `split.py` + catalogue | 2026-08-04 |

## The comparison table

One command, one split, one run: `python scripts/run_model.py --all --gallery`
(8m35s end to end on the Mac, 2026-08-04). Every model below was fitted on the same
1,136,199 train interactions and scored on the same 13,581 held-out books.

| Model | HitRate@10 | Coverage@10 | Novelty@10 | vs baseline | Ledger |
|---|---:|---:|---:|---|---|
| popularity (baseline) | 0.0145 | 0.019% | 10.81 | — | L22 |
| **item-item CF** | **0.0546** | 9.064% | 14.91 | **3.8× accuracy, 477× coverage** | L24 |
| ALS / weighted MF | 0.0451 | 0.835% | 12.63 | 3.1× accuracy, 44× coverage | L33 |
| item-item, explicit-only | 0.0379 | 10.739% | 16.59 | 2.6× accuracy | L26 |
| content TF-IDF | 0.0228 | 16.616% | 17.63 | 1.6× accuracy, 875× coverage | L30 |
| content embeddings | 0.0109 | **23.911%** | **18.42** | **0.75× accuracy**, 1,258× coverage | L35 |
| *structural ceiling* | *0.8481* | — | — | *no CF model can exceed this* | L20 |

**The table has a shape, and the shape is the finding.** Accuracy and coverage run in
opposite directions almost perfectly: the ranking by HitRate is exactly the reverse of
the ranking by Coverage, with ALS the only exception. No single model is best. Item-item
wins accuracy by a clear margin; embeddings reach twice the catalogue at a third of the
accuracy; and the content models are the only ones that can touch a book nobody has read.

**What we would actually ship, and why:** item-item as the scoring core, with the content
layer serving the catalogue it structurally cannot reach (L21: the union of both raises
the achievable ceiling from 84.8% to 95.4%). That is a hybrid recommended on measurements,
not on the fact that hybrids sound thorough.

**What none of these numbers prove.** Every row is one offline dataset, one split, one
proxy for a question the metric cannot answer: whether a reader would click. The ranking
is evidence about the models; it is not evidence about the product. A live A/B test is
the only thing that settles that, and it stays true no matter how favourable this table
looks.

## Baselines and models

All rows use the **L19** split (13,581 eligible users, seed 42) and identical metrics.
Command: `python scripts/run_model.py <name>`.

| ID | Model | HitRate@10 | Coverage@10 | Novelty@10 | Parameters | Measured |
|---|---|---|---|---|---|---|
| L22 | **popularity** (baseline) | **0.0145** | **0.019%** | 10.81 | rank by train interaction count, exclude own train items; `candidate_pool=2000` | 2026-08-04 |
| L24 | **item-item CF** (shrunk cosine) | **0.0546** | **9.064%** | 14.91 | binarized all-interaction matrix, shrinkage λ=10, 50 neighbours/item, min_support=1, score = Σ similarities over the user's train items | 2026-08-04 |
| L26 | item-item, **explicit-only ablation** | 0.0379 | 10.739% | 16.59 | identical model and parameters, fitted on the 420,090 graded interactions alone | 2026-08-04 |
| L33 | **ALS / weighted MF** | 0.0451 | 0.835% | 12.63 | `implicit` ALS, 128 factors, α=1 (confidence `1 + α·rating`), regularization 0.05, 20 iterations, seed 42 | 2026-08-04 |
| L35 | **content embeddings** (multilingual) | 0.0109 | **23.911%** | **18.42** | `paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, all 271,360 books embedded from title+author, profile vectors centered, score = mean cosine | 2026-08-04 |
| L30 | **content TF-IDF** (coverage layer) | 0.0228 | **16.616%** | 17.63 | char_wb 3–5-grams over title+author, min_df=3, 221,869 features, all 271,360 catalogue books vectorized, score = mean cosine to the user's train items | 2026-08-04 |

**L22 read out loud.** 197 of 13,581 users got their held-out book in a top-10 that was
essentially the same list for everybody. That is 491× better than ranking at random, and
it is exactly what ledger L9 predicts: when the top 1% of books hold 25.1% of all
interactions, guessing "bestseller" is a genuinely strong bet, and any model that cannot
beat 1.45% has learned nothing popularity did not already know. Read against the L20
ceiling of 84.8%, the baseline captures **1.7% of what is achievable** — so there is
plenty of headroom for a real model to claim.

The other half of the row is the warning. Coverage@10 = 0.019% means the baseline ever
recommends **51 distinct catalogue books** across all 13,581 users. It cannot sell the
long tail, which is where a bookseller's margin actually is. This is the failure mode
every later model is checked against: a HitRate that goes up while coverage stays near
zero is a bestseller list with extra steps.

*Aside worth keeping:* the single most-recommended book is *Wild Animus*
(2,501 train interactions), a novel its author gave away by the crate on BookCrossing.
The strongest signal in the raw popularity ranking is a marketing campaign, not a
reading preference — and the 7th entry is an ISBN with no catalogue metadata, so the
baseline recommends a book it cannot even name.

**L24 read out loud.** Item-item beats the baseline **3.8× on accuracy and 477× on
coverage** at the same time, which is the outcome that matters: it is not trading reach
for hit rate, it is better at both. In absolute terms it captures 6.4% of the L20
ceiling. It recommends 24,600 distinct catalogue books where the baseline manages 51.

**Where the baseline wins — honestly, nowhere on these three metrics**, and it is worth
saying that plainly rather than manufacturing a tie. It wins on *cost*: fit is instant
against 22 seconds, and the model is 52 items in memory rather than a 17M-entry
similarity matrix. It wins on *cold start*: for a brand-new user with no history the
baseline still returns a sensible list, where item-item returns nothing at all. Those are
real operational advantages and they are the reason a popularity fallback ships alongside
the model rather than being replaced by it.

**L26, the ablation — the pinned signal decision, now measured.** Fitting the identical
model on graded ratings alone costs **31% of the hit rate** (0.0546 → 0.0379). Discarding
the 62.3% of rows that carry no grade discards real predictive signal; an ungraded
interaction is weaker evidence than a 10, but it is not noise. Note the honest
counter-current: explicit-only scores *higher* on coverage (10.7% vs 9.1%) and novelty
(16.59 vs 14.91), because a sparser matrix spreads its recommendations more thinly. The
binarized signal wins where it counts and loses where it does not, and both directions
are in the row.

**L30 read out loud.** The content layer does what it was built for and fails where it
was expected to. It reaches **16.6% of the catalogue — 45,090 distinct books**, nearly
twice item-item's 24,600 and 880× the baseline's 51, and it is the only model that can
score a book nobody has touched. It also beats the baseline on accuracy (0.0228 vs
0.0145) while losing decisively to item-item (0.0546), which is the expected shape:
where collaborative evidence exists it is better evidence than a title, and the content
layer is there for the 95% of the catalogue where it does not exist.

**The two models are complementary rather than redundant**, which is the point of
proposing a hybrid: item-item reaches 29,733 distinct books, TF-IDF 45,090, and they
overlap on only 7,451. Together they touch **67,372 books — 24.8% of the catalogue**,
against 11.0% for item-item alone.

**L33 read out loud — and this row is a "no" that the project is better for having.** ALS
loses to item-item on all three metrics at once: less accurate (0.0451 vs 0.0546), and
its coverage is an order of magnitude worse (0.835% vs 9.064%, 2,365 distinct books). It
is also the most popularity-biased model in the table: **95.7% of its recommendations
come from the top 1% most-interacted books**, median train support 245.

That is exactly what the literature predicts for this dataset — on extreme sparsity,
well-regularized neighbourhood methods regularly beat matrix factorization, and the
popularity-bias study on Book-Crossing ([Naghiaei et al. 2022](https://arxiv.org/abs/2202.13446))
flags MF-family models as bias amplifiers. Measuring it here rather than citing it is the
difference between an opinion and a finding.

**ALS keeps its place in the story anyway, for reasons the metrics do not show.** Its item
factors give the single best item-to-item neighbourhoods of any model (see L34); the same
fit yields user factors, so personalization is free the day the product has user identity;
and it is the one model with a first-class Spark implementation, which makes the
productionization step a port rather than a rewrite. Recommending it as the *scoring core*
on this evidence would be wrong; dropping it from the ladder would be wrong too.

**L35 read out loud.** The embedding layer is the coverage extreme of the ladder:
**23.9% of the catalogue — 64,886 distinct books, 57,371 distinct works** — and the
highest novelty in the table, at the lowest accuracy, below even the popularity baseline.
That is the honest shape of a pretrained-embedding layer: it is a *coverage and
cold-start* mechanism, not a scoring core, and this project says so rather than promoting
it because it is the deep-learning component. Encoding all 271,360 books took 151s on the
Mac's GPU; vectors are cached under `artifacts/embeddings/` (gitignored, ~417 MB).

## Modelling decisions, measured rather than assumed

| ID | Decision | Measurement | Consequence | Measured |
|---|---|---|---|---|
| L23 | **No minimum-support threshold** (`min_support=1`) | Raising it to 5 keeps only 43,313 of 338,496 train items and drops the reachable share of held-out books from 84.8% to **64.5%** | 20 points of achievable accuracy spent to solve a problem shrinkage already handles continuously. The threshold stays off; shrinkage does the work | 2026-08-04 |
| L25 | **Shrinkage λ=10, 50 neighbours**, chosen on a *validation* split (`scripts/tune_item_item.py`, seed 43, 11,018 inner-eligible users) — never on the test holdout | λ=0 → HitRate 0.0296 / Coverage 17.5%; λ=10 → **0.0532 / 7.6%**; λ=100 → 0.0417 / 2.5%. Fewer neighbours beat more at every λ | Damping coincidental co-occurrence nearly doubles accuracy and costs more than half the catalogue reach. The accuracy/coverage tension in one table — and the reason the hybrid argument is made on coverage, not accuracy | 2026-08-04 |
| L27 | **The baseline is not a weak accuracy benchmark — it is a *narrow* one** | HitRate by held-out book's train support: 0 interactions **0.0000**, 1–4 **0.0000**, 5–49 **0.0000**, 50+ **0.0531**. Item-item over the same strata: 0.0000 / 0.0170 / 0.0521 / 0.1163 | The baseline's entire 1.45% comes from the 3,713 users whose target was already a bestseller. It contributes *exactly nothing* for the other 73%. Aggregate HitRate hides this completely | 2026-08-04 |
| L28 | *Negative result:* **item-item degrades on long user profiles** | HitRate by train-profile length: 0–9 items **0.0607**, 10–24 **0.0610**, 25–74 0.0557, 75+ **0.0308** | Summing similarities over a long profile lets volume drown the signal — a known item-KNN weakness we did not correct. The fix (normalize by profile length, or score from the user's strongest *n* items) is a concrete next step, not a mystery | 2026-08-04 |
| L31 | *Negative result, and the most actionable finding so far:* **a third of the content model's output is the same book again** | Share of recommended slots that are another *edition* (same normalized title+author, ledger L15) of a book already in the user's train profile: **TF-IDF 31.6%**, item-item 0.7%. **73.4%** of users get at least one such recommendation from TF-IDF, against 5.1% from item-item. Gallery: *The Da Vinci Code*'s top 7 content neighbours are 7 ISBNs of *The Da Vinci Code*; *Harry Potter and the Sorcerer's Stone*'s top 8 are 8 editions of itself | A text model cannot tell "same work, different ISBN" from "similar book", because the two are textually identical. Nearly a third of TF-IDF's top-10 is therefore unusable output, and its coverage advantage is partly an artefact of counting editions as distinct books. **Edition clustering in the data-prep layer is not a nice-to-have; it is the difference between a demo that works and one that recommends the book the user is holding.** Deliberately not patched yet: it changes the comparison basis for every model, so it is a decision to take before the next full run | 2026-08-04 |
| L32 | **Coverage is inflated by edition duplication for every model** | Measuring distinct *works* instead of distinct ISBNs: TF-IDF 16.62% → **13.46%**, item-item 10.96% → **8.81%** | The ranking between models survives, the absolute numbers shrink by about a fifth. Per-ISBN coverage is the number reported in the table above; per-work is the more honest one to quote, and both are here so either can be defended | 2026-08-04 |
| L34 | **ALS item-similarity needs a support floor, and with one it gives the best neighbourhoods of any model here** | 196,054 of 338,496 train items were touched exactly once; their factors are noise directions with mean norm 0.07 against 1.35 for items with 50+ interactions. With ~196k of them, the best chance alignment in 128 dimensions reaches cosine 0.95. **Unfiltered**, *Harry Potter*'s nearest neighbours were five one-reader books tied at 0.941. **With a floor of 20**: *The Fellowship of the Ring*, then Harry Potter 3, 2 and 4. *The Da Vinci Code* → *Angels & Demons*, *Digital Fortress*, *Deception Point* — all Dan Brown | Same factors, same formula: the noise simply outnumbered the signal at the argmax. Fixed by requiring 20 train interactions on the *similarity* endpoint only; `recommend` and every metric above are untouched. This solves L29's failure mode, on the model L33 says is otherwise the weakest — the ladder's rungs are good at different things | 2026-08-04 |
| L36 | **Dense profile vectors collapse; centering fixes it** | Mean cosine of a user's averaged profile vector to the *global* profile centroid: **0.883** — every user's profile points almost the same way, so the model recommends one generic region to everybody. Item vectors themselves: 0.518. Subtracting the global mean and renormalizing drops collapse to **0.193**, and on validation (seed 43) lifts HitRate 0.0036 → **0.0095** and Coverage 3.7% → **20.4%** | Sentence embeddings share a large common component; averaging amplifies it. This is why the naive "embed everything and take cosine" recipe underperforms — and why the fix is one line once the diagnosis is right. Chosen on validation, never on test | 2026-08-04 |
| L37 | **The two product paths need different geometry** | Free-text lookup over 7 queries: centering pushed the right book from rank 1→4 (*el senor de los anillos*), 2→5 (*harry potter stein*), 3→4 (*da vinci code*); both variants found 5/7 in the top five | A lookup query *is* a point, not an average, so the common direction is part of what matches it to a title. `find_book` therefore serves from the uncentered vectors while `recommend` uses the centered ones. Two paths, two geometries, both measured | 2026-08-04 |
| L38 | *Negative result:* **cross-lingual lookup does not work on titles this short** | `"der kleine prinz"` → correct at rank 1; `"lovely bones sebold"` → rank 1; `"el senor de los anillos"` → rank 1. But `"herr der ringe"` and `"hobit tolkien"` return nothing relevant in the top 5, and `"harry potter stein"` is beaten to rank 1 by *Hoopla — Harry Stein* | A multilingual encoder bridges *sentences*; title+author is three to five words, too thin a signal for German→English transfer. This is the doc's own stated limit, now measured — and it is the concrete argument for an LLM metadata-enrichment layer (generating genre tags, themes and a short description from title+author): more text per book is exactly what would fix it | 2026-08-04 |
| L39 | **Both content models' item-to-item surfaces are unusable without edition clustering** | Top-5 neighbours of each anchor under embeddings: *The Da Vinci Code* → 5 ISBNs of *The Da Vinci Code*; *Harry Potter* → 5 editions of itself; *The Lovely Bones* → 5 editions of itself. Recommendation slots that duplicate a book the user already has: embeddings **8.2%**, TF-IDF 31.6%, item-item 0.7% | Text models cannot distinguish "same work, different ISBN" from "similar book" — they are textually identical. Centering happens to reduce the duplicate rate in *recommendations* (8.2% vs TF-IDF's 31.6%) but does nothing for the *similarity* endpoint. Edition clustering (L15) is a precondition for shipping either content model as the app's similarity engine | 2026-08-04 |
| L29 | *Negative result:* **the item-to-item surface degrades at medium support, where the offline metric cannot see it** | Face-validity gallery: for *The Da Vinci Code* (853 interactions) the top neighbour is *Angels & Demons*, same author; for *The Lovely Bones* (1,248) it is *Lucky: A Memoir*, same author. For *Harry Potter and the Sorcerer's Stone* (101) the top two neighbours are unrelated obscure books that share 4 readers out of 6, scoring 0.116 against *Chamber of Secrets* at 0.097 | λ=10 was selected for HitRate, which is dominated by popular held-out items, so it is under-damped for the item-to-item product surface. **This is the gap between the offline harness and the product surface, as a concrete measurement rather than a caveat.** Options: a minimum co-occurrence floor, a higher λ for the similarity endpoint than for ranking, or the content layer carrying mid-support anchors | 2026-08-04 |

Metric definitions, identical for every row (`src/recommender/eval.py`):
**HitRate@10** — share of eligible users whose held-out book is in their top-10. Under
leave-one-out this equals Recall@10, and Precision@10 = HitRate@10 / 10; one number,
three names. **Coverage@10** — distinct catalogue books appearing in any user's top-10,
over all 271,360; recommended ISBNs absent from `Books.csv` do not count, because a book
we cannot name is a book we cannot show. **Novelty@10** — mean
`-log2((train_interactions + 1) / (total_train_interactions + 271,360))`; the +1
smoothing keeps the metric finite for the zero-interaction books only content models can
reach.

## Open items this ledger will need

- Edition clustering in the data-prep layer (L31, L39), and the full comparison re-run on
  the clustered catalogue — it moves the coverage number of every model.
- The hybrid itself measured, rather than argued from the L21 ceiling: item-item scoring
  core with the content layer serving what it cannot reach.
- Item-item normalized by profile length, to see whether it removes the long-profile
  degradation in L28.

---

**Honesty note that travels with every number above.** These are offline,
single-dataset measurements on a 2004 crawl. They describe the data, and later they will
describe model behaviour on a held-out slice of it. They are a proxy for whether
recommendations are *good* — a live A/B test is the only real proof, and that stays true
however favourable the offline table looks.
