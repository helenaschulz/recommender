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
| L15 | The same work is split across editions — **a lower bound, superseded by L40** | 17,554 works over 40,675 ISBNs (15.0% of the catalogue) | `groupby` on lower-cased, whitespace-stripped title+author with more than one distinct ISBN. On raw strings it is 15,746 / 35,921 — the normalization matters and is part of the claim. An *exact* author string cannot see "Fyodor" / "Fedor" / "Fyodor M." as one person, so this counts fewer duplicates than exist; **L40 measures 59,928 ISBNs, 47% more** | §5 | 2026-08-03 |
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
| L44 | **item-item CF, work level** (M11) | **0.0644** | 8.190%* | 14.17 | identical model and parameters, fitted on the **work-keyed** matrix: 235,824 items, 13,580 eligible users. *\*Coverage is over 235,824 works, not 271,360 ISBNs — this row is **not** cell-comparable with L24* | 2026-08-08 |
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
ceiling. It recommends 24,597 distinct catalogue books where the baseline manages 51.

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
twice item-item's 24,597 and 880× the baseline's 51, and it is the only model that can
score a book nobody has touched. It also beats the baseline on accuracy (0.0228 vs
0.0145) while losing decisively to item-item (0.0546), which is the expected shape:
where collaborative evidence exists it is better evidence than a title, and the content
layer is there for the 95% of the catalogue where it does not exist.

**The two models are complementary rather than redundant**, which is the point of
proposing a hybrid: item-item reaches 24,597 distinct catalogue books, TF-IDF 45,090, and
they overlap on only 7,451. Together they touch **62,236 books — 22.9% of the catalogue**
(49,110 distinct works, 20.8%), against 9.064% for item-item alone.

*(Corrected 2026-08-08, ledger L46. This paragraph previously read "29,733 … 67,372 books
— 24.8% … against 11.0%". Those item-item figures counted every recommended ISBN,
including the ones with no row in `Books.csv`, while the TF-IDF figure beside them counted
only catalogue books. Two denominators, one sentence.)*

**L33 read out loud — and this row is a "no" that the project is better for having.** ALS
loses to item-item on all three metrics at once: less accurate (0.0451 vs 0.0546), and
its coverage is an order of magnitude worse (0.835% vs 9.064%, 2,267 distinct catalogue books). It
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
**23.9% of the catalogue — 64,886 distinct books, 54,494 distinct works** — and the
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
| L32 | **Coverage is inflated by edition duplication for every model** — *numbers superseded by L46* | Measuring distinct *works* instead of distinct ISBNs: TF-IDF 16.62% → **14.21%**, item-item 9.064% → **9.92%** | The ranking between models survives; per-work coverage is the more honest number to quote and both are here so either can be defended. **The original row said TF-IDF 13.46% and item-item 10.96% → 8.81%.** Both were wrong, in two different ways: item-item's ISBN figure counted non-catalogue ISBNs (L46), and the per-work percentages divided distinct *works* by the 271,360-ISBN catalogue instead of by the 235,824 works. Against the right denominator, per-work coverage is *higher* than per-ISBN coverage for item-item, not lower | 2026-08-08 |
| L34 | **ALS item-similarity needs a support floor, and with one it gives the best neighbourhoods of any model here** | 196,054 of 338,496 train items were touched exactly once; their factors are noise directions with mean norm 0.07 against 1.35 for items with 50+ interactions. With ~196k of them, the best chance alignment in 128 dimensions reaches cosine 0.95. **Unfiltered**, *Harry Potter*'s nearest neighbours were five one-reader books tied at 0.941. **With a floor of 20**: *The Fellowship of the Ring*, then Harry Potter 3, 2 and 4. *The Da Vinci Code* → *Angels & Demons*, *Digital Fortress*, *Deception Point* — all Dan Brown | Same factors, same formula: the noise simply outnumbered the signal at the argmax. Fixed by requiring 20 train interactions on the *similarity* endpoint only; `recommend` and every metric above are untouched. This solves L29's failure mode, on the model L33 says is otherwise the weakest — the ladder's rungs are good at different things | 2026-08-04 |
| L36 | **Dense profile vectors collapse; centering fixes it** | Mean cosine of a user's averaged profile vector to the *global* profile centroid: **0.883** — every user's profile points almost the same way, so the model recommends one generic region to everybody. Item vectors themselves: 0.518. Subtracting the global mean and renormalizing drops collapse to **0.193**, and on validation (seed 43) lifts HitRate 0.0036 → **0.0095** and Coverage 3.7% → **20.4%** | Sentence embeddings share a large common component; averaging amplifies it. This is why the naive "embed everything and take cosine" recipe underperforms — and why the fix is one line once the diagnosis is right. Chosen on validation, never on test | 2026-08-04 |
| L37 | **The two product paths need different geometry** | Free-text lookup over 7 queries: centering pushed the right book from rank 1→4 (*el senor de los anillos*), 2→5 (*harry potter stein*), 3→4 (*da vinci code*); both variants found 5/7 in the top five | A lookup query *is* a point, not an average, so the common direction is part of what matches it to a title. `find_book` therefore serves from the uncentered vectors while `recommend` uses the centered ones. Two paths, two geometries, both measured | 2026-08-04 |
| L38 | *Negative result:* **cross-lingual lookup does not work on titles this short** | `"der kleine prinz"` → correct at rank 1; `"lovely bones sebold"` → rank 1; `"el senor de los anillos"` → rank 1. But `"herr der ringe"` and `"hobit tolkien"` return nothing relevant in the top 5, and `"harry potter stein"` is beaten to rank 1 by *Hoopla — Harry Stein* | A multilingual encoder bridges *sentences*; title+author is three to five words, too thin a signal for German→English transfer. This is the doc's own stated limit, now measured — and it is the concrete argument for an LLM metadata-enrichment layer (generating genre tags, themes and a short description from title+author): more text per book is exactly what would fix it | 2026-08-04 |
| L39 | **Both content models' item-to-item surfaces are unusable without edition clustering** | Top-5 neighbours of each anchor under embeddings: *The Da Vinci Code* → 5 ISBNs of *The Da Vinci Code*; *Harry Potter* → 5 editions of itself; *The Lovely Bones* → 5 editions of itself. Recommendation slots that duplicate a book the user already has: embeddings **8.2%**, TF-IDF 31.6%, item-item 0.7% | Text models cannot distinguish "same work, different ISBN" from "similar book" — they are textually identical. Centering happens to reduce the duplicate rate in *recommendations* (8.2% vs TF-IDF's 31.6%) but does nothing for the *similarity* endpoint. Edition clustering (L15) is a precondition for shipping either content model as the app's similarity engine | 2026-08-04 |
| L29 | *Negative result:* **the item-to-item surface degrades at medium support, where the offline metric cannot see it** | Face-validity gallery: for *The Da Vinci Code* (853 interactions) the top neighbour is *Angels & Demons*, same author; for *The Lovely Bones* (1,248) it is *Lucky: A Memoir*, same author. For *Harry Potter and the Sorcerer's Stone* (101) the top two neighbours are unrelated obscure books that share 4 readers out of 6, scoring 0.116 against *Chamber of Secrets* at 0.097 | λ=10 was selected for HitRate, which is dominated by popular held-out items, so it is under-damped for the item-to-item product surface. **This is the gap between the offline harness and the product surface, as a concrete measurement rather than a caveat.** Options: a minimum co-occurrence floor, a higher λ for the similarity endpoint than for ranking, or the content layer carrying mid-support anchors | 2026-08-04 |

## Edition clustering (milestone M11)

The problem was found in M7 and left unpatched on purpose (L31): a third of the content
model's output is another edition of a book the reader already has, and it re-bases every
coverage number in the table. M11 measures it, then acts on it. Source for every line
below: `python scripts/analyze_editions.py` and `python scripts/analyze_dedup.py`,
clustering implemented in `src/recommender/data.py` and pinned by `tests/test_works.py`.

**The key**, recorded once: the title with its trailing parenthetical stripped and
normalized (HTML-unescaped, lower-cased, whitespace collapsed), plus the author's
last-name token. Plus one extension — see L41.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L40 | **The catalogue is 13% smaller than its ISBN count** | 271,360 ISBNs → **235,824 works**; 24,392 works hold more than one ISBN, covering **59,928 ISBNs (22.1%)** | Clustering on the key above. Largest cluster: *Little Women* / Alcott, 53 ISBNs. **This supersedes L15**, which used an exact title+author string and found 40,675 duplicated ISBNs; this key finds **47% more**, because an exact author string cannot see the spelling variants | 2026-08-08 |
| L41 | **The pinned key does not merge Dostoevsky with Dostoyevsky; one narrow extension does** | **223 clusters** change; duplicated ISBNs 59,582 → **59,928 (+346, 0.6%)** | The pinned key is title + surname token, which absorbs Fyodor/Fedor/Feodor/Fyodor M. but not a surname respelled. Extension: within an *identical* normalized title, surnames within one edit and ≥6 characters merge, canonical = the most frequent spelling. `cluster_works(merge_author_variants=False)` reproduces the pinned key exactly, which is how this row is measured | 2026-08-08 |
| L42 | **Sample validation: 0 wrong merges in 30, 1 in the 20 that test the extension** | random sample **0/30**; transliteration sample **1/20** | 30 seeded-random multi-ISBN clusters (seed 42) inspected by hand in [`edition_clusters_sample.md`](edition_clusters_sample.md). Because the extension touches only 223 of 24,392 clusters, a uniform draw cannot audit it, so 20 of *those* were drawn separately. The one error: Anne Hampson and Georgia Hampton each wrote a *Desire*; `hampson`/`hampton` is one edit at length 7. Cost of removing it — a floor of 8 — is also losing Rendell/Rendall, Elliott/Elliot, Searls/Searles and Higgins/Higgns, so it stays and is reported | 2026-08-08 |
| L43 | *Negative result of the standard recipe:* **min-5 filtering silently deletes editions of books that clear the threshold** | **23,429 ISBNs** carrying **49,649 interactions** sit below min-5 while belonging to a work that clears it | Per-ISBN interaction counts against per-work totals over all 1,149,780 interactions. *Crime and Punishment* is the case in one line: 21 ISBNs, 141 interactions, strongest edition 40, **13 editions below 5 carrying 27 interactions**. Filtering at ISBN level throws away **19.1%** of that novel's evidence and calls what is left a book with 40 readers | 2026-08-08 |
| L44 | **Clustering lifts the structural ceiling (L20) slightly, and item-item's accuracy a lot more** | ceiling 84.81% → **86.66%**; item-item HitRate **0.0546 → 0.0644 (+18%)** | Same split mechanics on work ids: per-user leave-one-out, seed 42, relevance ≥8, **13,580 eligible users** (one user loses eligibility when their graded editions collapse). Coverage@10 8.190% is measured against 235,824 works, not 271,360 ISBNs — the two coverage cells are **not** comparable. Share of the achievable ceiling captured: 6.44% → **7.43%**. Runtime 31s | 2026-08-08 |

**L44 read out loud — the one number in this milestone that changes a recommendation.**
Merging editions before training buys **+18% relative on HitRate from a data-prep change,
with no model change at all**. For scale, the whole gap between the best and second-best
model in the comparison table — item-item 0.0546 against ALS 0.0451 — is 21%. The
mechanism is the one Helena's analysis predicted: co-occurrence counts that were split
across 21 ISBNs of *Crime and Punishment* become one count, so evidence that shrinkage was
correctly suppressing as coincidence is now large enough to survive it.

It is also a **lower bound on what work-level modelling is worth**: λ=10 and 50 neighbours
were tuned on the ISBN-level validation split (L25) and were reused unchanged, so the
work-level model is running on someone else's hyperparameters. Re-tuning on a work-level
validation split is the obvious next measurement, and it can only help.

Three checks before believing it, because a result this clean is a bug first:

1. **Is it just an easier target?** The ceiling moves 84.81% → 86.66%, which is +2.2%
   relative. The HitRate moves +18%. The lift survives the denominator.
2. **Did a held-out edition leak in through a second edition the same user owns?**
   Impossible by construction — `to_work_level` collapses each (user, work) pair *before*
   the split, so holding out a work removes every edition of it from that user's profile
   (pinned by `tests/test_works.py`). And at ISBN level this path was worth almost
   nothing anyway: only **51 of 13,581 holdouts (0.38%)** were a second edition of
   something the user already had, so the ISBN-level baseline was not being flattered.
3. **Same code path?** Yes, deliberately: work ids are written into the `ISBN` column and
   the identical split, matrix builder and metrics run over them. There is no second
   implementation to disagree with the first.

The honest limits: the item universe is different, so the two rows are **not comparable
cell by cell** — 0.0644 answers "does edition clustering lift accuracy", not "is this
model better than L24's". Coverage *falls* in percentage terms (9.064% of ISBNs → 8.190%
of works), which is arithmetic rather than a regression: collapsing editions removes
duplicated items from the numerator and the denominator at different rates. And the
clustering itself is the measured-imperfect thing in L42.

### Does stripping the parenthetical ever merge two different books?

Raised by Helena on 08.08 from the §8 gallery: the trailing parenthetical is not always a
series — *(Book 4)*, *(Trophy Newbery)*, *(3rd Edition)*, *(Paperback)*. If the
parenthetical were the only thing separating two volumes, stripping it would merge two
different works. Measured rather than argued: `python scripts/analyze_editions.py` §6.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L48 | **The `series` field is misnamed, and the merge risk it implies is 0.023%** | of 74,233 parentheticals: **27.1% volume/part numbers**, 6.7% format, 1.1% numbered editions, the rest series names, imprints and awards. Clusters whose members carry *contradictory* numbers: **100 of 24,392** (1,648 of 480,857 merged interactions, 0.34%); genuinely different books among them: **19 clusters, 113 interactions, 0.023%** | Parenthetical text classified by regex; a cluster is flagged when two members' parentheticals contain different digits, and separately when both contain an ordinal edition (`Nth ed`). The feared collision — *(Book 1)* merging with *(Book 2)* — **does not occur in this catalogue, because the volume is carried by the title** (*Harry Potter and the Goblet of Fire*), not only by the parenthetical. Almost all "contradictory numbers" are publisher catalogue numbers that differ between reissues of the *same* book (*Twilight Magic (Harlequin American Romance, No 16504)* vs *(No. 504)*), where merging is correct | 2026-08-08 |

**L48 read out loud.** The residue splits in two, and only half of it is an error. Textbook
and handbook revisions — *MLA Handbook* 5th and 6th, *Programming Perl* 2nd and 3rd,
*Business* 5th and 6th — arguably *should* merge: a reader asking for *Programming Perl*
wants the book, not an edition. Annual and serial guides should not: *Lonely Planet
Portugal* 2nd and 3rd, *Frommer's Colorado* 4th and 6th, and *Schroeder's Antiques Price
Guide* 15th and 19th are different books with different contents, and those are true false
merges. That subset is roughly a third of 19 clusters and about 20 interactions.

**Not patched, deliberately.** A rule keyed on ordinal editions would separate the travel
guides and simultaneously split the textbook cases where merging is right — a wash, bought
with a special case that has to be explained. At 0.023% of merged interactions it is not
worth the rule; it is worth the line in this ledger. The naming is the other half: the
field is called `series` because that is its most common content, but it holds edition
packaging generally. Renaming it (`edition_note`?) is a one-line change if preferred.

### Deduplication at serving time, and the counting basis

One run, all four models, `python scripts/analyze_dedup.py` (17m12s on the Mac,
2026-08-08, all 13,581 users — inside the runtime guardrail, no sampling). Each model is
scored twice, at k=10 and at k=100: the k=10 pass reproduces its comparison-table row
exactly (0.0546, 0.0228, 0.0109, 0.0451 — all four match L24/L30/L35/L33 to the digit),
which is the check that the "before" column is the ledger's own number and not a
re-derivation of it.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L45 | **Serving-time dedup removes the duplicate output at no accuracy cost — and TF-IDF gains 21%** | duplicate slots → **0.0%** for every model; HitRate: TF-IDF **0.0228 → 0.0277**, item-item 0.0546 → 0.0546, ALS 0.0451 → 0.0454, embeddings 0.0109 → 0.0108 | `WorkDeduped` (`src/recommender/serving.py`) asks the model for 100 candidates and keeps the first per work, skipping works the user already owns. Duplicate-slot rates *before* dedup, on the M11 key: **TF-IDF 39.1% of slots / 81.5% of users**, embeddings 11.3% / 38.5%, ALS 1.9% / 11.8%, item-item 1.2% / 7.9%. These are higher than L31/L39 (31.6% / 8.2% / 0.7%) because the better key finds duplicates the exact-string key could not. Lists still fill: 99.4–100% of slots occupied | 2026-08-08 |
| L46 | **The counting-basis fix: Coverage@10 was reported on two different denominators** | item-item **24,597 catalogue ISBNs (9.064%)**, not 29,733 (10.96%); complementarity union **62,236 (22.9%)**, not 67,372 (24.8%) | The pinned basis is the one `eval.py` has always used and the one every table cell reports: *distinct recommended ISBNs that exist in `Books.csv`, over 271,360*. A book we cannot name is a book we cannot show. The complementarity paragraph and L32 had instead counted **all** recommended ISBNs for item-item — including the ones with no catalogue row — while quoting TF-IDF on the catalogue basis in the same sentence. TF-IDF and embeddings were never affected: their candidate universe *is* `Books.csv`, so the two bases coincide (45,090 and 64,886 either way). ALS was: 2,267 catalogue, 2,365 total. Per-work coverage against the 235,824-work denominator: item-item **9.92%**, TF-IDF **14.21%**, embeddings **23.11%**, ALS **0.88%** | 2026-08-08 |

**L45 read out loud.** The headline is that this is **free**. Four in five TF-IDF users
were being shown a book they already had; removing those slots does not cost accuracy, it
*buys* 21% of it, because a wasted slot gets refilled with a real candidate. Item-item and
ALS barely move, which is the expected shape — collaborative similarity already separates
editions, since two ISBNs of one book are read by different people and so do not
co-occur. Embeddings lose 0.0001, about one user in 13,581: noise, not a cost.

**And the important caveat, because 0.0% is too clean to take at face value.** The
duplicate rate is measured with the *same* work key that performed the deduplication, so
it is guaranteed to be zero and is **not independent evidence**. The independent check is
the gallery, and the gallery is less flattering: after dedup, TF-IDF's top neighbour for
*The Da Vinci Code* is still *El Codigo Da Vinci / The Da Vinci Code*, and the embedding
model answers *Harry Potter and the Sorcerer's Stone* with the Philosopher's Stone, the
French, Spanish, Italian and German editions. Those are the same work under different
*titles*, and a key built on title equality cannot see them by construction. See L47.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L47 | *Negative result:* **dedup cleans the collaborative surfaces completely and the content surfaces only partly** | Same-work neighbours surviving dedup, out of 30 gallery slots per model: item-item **0/30**, ALS **0/30**, TF-IDF **7/30**, embeddings **9/30** | Hand count over the deduped 3-anchor gallery (3 anchors × top-10), the same method as L29/L34/L39. A slot counts as a survivor if it is the anchor's own text under a different title: a translation (*Desde Mi Cielo*, *Harry Potter und der Stein der Weisen*), a subtitle variant (*The Lovely Bones* vs the anchor's *The Lovely Bones: A Novel*), an alternate regional title (*Philosopher's Stone*), a dual-language title (*El Codigo Da Vinci / The Da Vinci Code*), or a re-credit to the illustrator (*Mary Grandpre*). Sequels and books *about* the anchor do not count — they are legitimately similar. Worst single case: embeddings on *Harry Potter*, **7 of 10** | 2026-08-08 |

**L47 read out loud — this is the honest ceiling on what M11 achieved.** Edition
clustering solved the problem it could solve: ISBNs of a work that share a title. It
cannot solve the problem underneath, which is that *Harry Potter and the Philosopher's
Stone*, *Harry Potter a l'ecole des sorciers* and *Harry Potter und der Stein der Weisen*
are one book with three names, and no amount of string normalization will discover that
from title+author alone. The collaborative models were never affected — two ISBNs of one
book are read by *different* people, so they do not co-occur, and item-item and ALS score
a clean 0/30. It is exactly the text-based models that stay broken.

**Which makes this the strongest argument the project has for the enrichment layer.** L38
found the same wall from the other side: cross-lingual *lookup* fails because title+author
is three to five words, too thin for a multilingual encoder to bridge. Both failures have
the same fix and it is not more string processing — it is more text per book (LLM-generated
descriptions, themes, genre tags) or an external work identifier that already knows these
are one book. That is a Part 3 proposal with two measurements behind it rather than an
opinion about LLMs being useful.


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

- ~~Edition clustering in the data-prep layer (L31, L39)~~ — **done, M11 (L40–L47).** What
  remains is Helena's decision: does the whole comparison table move to work level? L44
  says it is worth +18% on the one model measured both ways; the other five have not been
  re-run at work level, and the content models cannot be until the catalogue itself is
  work-keyed.
- **Re-tune item-item on a work-level validation split** (L44 used ISBN-level λ and
  neighbourhood size, so the +18% is a lower bound).
- **Beyond title equality** (L47): translations and alternate titles still defeat the
  clustering, and only the text-based models suffer. Either LLM metadata enrichment or an
  external work identifier; both also address the lookup failure in L38.
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
