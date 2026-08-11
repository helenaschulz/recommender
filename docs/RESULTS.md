# RESULTS.md — the measurement ledger

Every number that appears anywhere in this project — a README claim, a notebook takeaway,
a statement like "beats the baseline by X" — traces to a line in this file. **New claim,
new line**, including the negative results. If a number is not here, it does not get
quoted.

Each line records *how* it was measured, not just what came out, so any number can be
re-derived or challenged. `L1.` … numbering is stable; everything else cites line IDs.

Column meanings: **Source** is the artefact that produced the number (notebook section,
script, or run). **Measured** is the date it was last recomputed.

Where a line's reasoning is too long to sit in a table cell, the cell keeps the claim, the
number and the method, and the discussion follows the table as a paragraph headed with that
line's ID. Nothing is dropped in the move: a ledger entry that has been shortened is one
that can no longer be checked.

**This file is a reference, not a read-through.** For the argument in two pages see
[`SUMMARY.md`](SUMMARY.md); for what was built in what order, [`change_log.md`](change_log.md).

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
| L7 | Most books are rated exactly once | **57.9%** (87.1% have fewer than 5); on the **explicit** subset **69.7%** (92.2% under 5) | Share of `groupby("ISBN").size()` equal to 1 (resp. < 5). The explicit-only pair repeats the same computation on `Book-Rating > 0` and is the honest number to quote whenever the sparsity argument is made about *graded* data — it was quoted in `dataset_findings.md` before it had a line here, and is recomputed directly from `Ratings.csv` (57.8598 / 87.1490 / **69.6988** / 92.1843) | §3 | 2026-08-09 |
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

## The primary comparison table — one row per *work*

One command, one split, one run: `python scripts/run_model.py --all --gallery --work-level`
(~8 min end to end on the Mac, 2026-08-08). Every model below was fitted on the same
1,129,755 train interactions over **235,824 works** and scored on the same 13,580 held-out
works. **The item is a work, not an ISBN** — milestone M12, defined in L49, and the reason
the whole table moved is measured in L58.

| Model | HitRate@10 | Coverage@10 | Novelty@10 | vs baseline | Ledger |
|---|---:|---:|---:|---|---|
| popularity (baseline) | 0.0155 | 0.027% | 10.54 | — | L52 |
| **item-item CF** | **0.0644** | 8.190% | 14.17 | **4.2× accuracy, 302× coverage** | L53 |
| ALS / weighted MF | 0.0545 | 0.897% | 12.29 | 3.5× accuracy, 33× coverage | L55 |
| item-item, explicit-only | 0.0486 | 10.036% | 15.97 | 3.1× accuracy, 370× coverage | L54 |
| content TF-IDF | 0.0405 | 16.806% | 17.07 | 2.6× accuracy, 619× coverage | L56 |
| content embeddings | 0.0141 | **26.143%** | **18.34** | **not measurably different from the baseline** (0.91×; McNemar p = 0.342, L74), 963× coverage | L57 |
| *structural ceiling* | *0.8666* | — | — | *no CF model can exceed this* | L50 |

Coverage ratios are taken against the baseline's **64 distinct works**, the exact count;
every percentage in the column is over the same 235,824-work denominator.

**The table has a shape, and the shape is the finding.** Among the four real models,
accuracy and coverage run in opposite directions: ordered by HitRate they are item-item >
ALS > explicit-only > TF-IDF; ordered by Coverage that reverses exactly, TF-IDF >
explicit-only > item-item > ALS. **The baseline is outside that trade-off rather than at
one end of it** — it is last on accuracy among the real models *and* last on coverage, so
it buys nothing in either direction and only marks the zero point. Embeddings sit at the
coverage extreme at an accuracy the baseline matches (see the sampling-error note below).
No single model is best. Item-item wins accuracy by a clear margin; embeddings reach three
times the catalogue at a fifth of the accuracy; and the content models are the only ones
that can touch a book nobody has read. **That shape survived the re-base unchanged** —
same ordering, same tension, a different item universe — which is the reassuring half of
M12.

*(Corrected 2026-08-09. This paragraph previously claimed the HitRate ranking was "exactly
the reverse" of the Coverage ranking "with ALS the only exception". It is not: the
popularity baseline is fifth on accuracy and sixth on coverage, so it is displaced by four
places, and item-item by three. The trade-off is real and holds among the four real models;
the baseline was never part of it. The table above always said so — the sentence did not.)*

**How large a difference in this table is real?** Not measured, *derived*, and labelled as
such: with 13,580 users and leave-one-out, HitRate is a binomial proportion, so its standard
error is `sqrt(p(1-p)/13580)` — **±0.0010 to ±0.0021** across these rows. A difference of
0.0099 (item-item over ALS) is roughly 3.5 standard errors even on the conservative unpaired
assumption and is safe to call. **The difference between the popularity baseline (0.0155)
and content embeddings (0.0141) is 19 users out of 13,580, z ≈ 1.0, and is not
distinguishable from noise.** That row previously read "0.91× accuracy" and L57 read "still
below the baseline"; both were reworded on 2026-08-09 to say what the sample supports, which
is that the two are not measurably different. The ISBN-level analogue was genuinely
different — 0.0109 against 0.0145 is z ≈ 2.7 — so the re-base moved embeddings *up* into the
noise band rather than the wording having always been wrong. Every other gap in the table
exceeds three standard errors. **The proper test is paired, and it has now been run: L74
measures McNemar over the per-user hit vectors and agrees with this derivation** — embeddings
against the baseline is 190 wins to 210 losses, p = 0.342, and every other comparison in the
table is distinguishable, the narrowest of them (ALS against item-item) at p = 2.7e-06. The
95% interval on each cell is in L74; they run **±0.002 to ±0.004**.

**What we would actually ship, and why — and this paragraph is now a measurement rather
than a bound.** Item-item as the scoring core, with the content layer as a **backfill**:
where item-item cannot fill ten slots, TF-IDF fills the rest. **Measured (L76): HitRate@10
0.0644 → 0.0650, Coverage@10 8.190% → 8.529%, nine readers gained and none lost, paired
p = 0.0039.** That is the whole of it, and it is small — the coverage argument this project
has made since M10 (L50: the union raises the achievable ceiling from 86.7% to 95.3%) is
worth about half a percent of item-item's hits once it has to be *ranked* rather than merely
reached. Two rules score higher — score fusion 0.0690, RRF 0.0687 — and **neither is
recommended**, because L79 measures what they cost: a third of readers would see another
edition of a book they already own, and the accuracy is taken from well-evidenced readers
rather than added. The hybrid is real, it is the modest version of itself, and saying so is
the finding.

**What none of these numbers prove.** Every row is one offline dataset, one split, one
proxy for a question the metric cannot answer: whether a reader would click. The ranking
is evidence about the models; it is not evidence about the product. A live A/B test is
the only thing that settles that, and it stays true no matter how favourable this table
looks.

### The ISBN-level table — the journey record, kept on purpose

This was the primary table from M4 to M11 and is **not** deleted, for two reasons: it is
how the model choice was actually reached, and the distance between it and the table above
is itself a finding (L58). It is measured on 271,360 ISBNs and 13,581 held-out books, so
**no cell here is comparable with a cell above**: different items, different denominator,
different eligibility.

One command, one split, one run: `python scripts/run_model.py --all --gallery` (8m35s,
2026-08-04).

| Model | HitRate@10 | Coverage@10 | Novelty@10 | vs baseline | Ledger |
|---|---:|---:|---:|---|---|
| popularity (baseline) | 0.0145 | 0.019% | 10.81 | — | L22 |
| **item-item CF** | **0.0546** | 9.064% | 14.91 | **3.8× accuracy, 477× coverage** | L24 |
| ALS / weighted MF | 0.0451 | 0.835% | 12.63 | 3.1× accuracy, 44× coverage | L33 |
| item-item, explicit-only | 0.0379 | 10.739% | 16.59 | 2.6× accuracy | L26 |
| content TF-IDF | 0.0228 | 16.616% | 17.63 | 1.6× accuracy, 875× coverage | L30 |
| content embeddings | 0.0109 | **23.911%** | **18.42** | **0.75× accuracy**, 1,258× coverage | L35 |
| *structural ceiling* | *0.8481* | — | — | *no CF model can exceed this* | L20 |

## Baselines and models — the ISBN-level rows (M5–M10)

All rows use the **L19** split (13,581 eligible users, seed 42) and identical metrics, on
the **ISBN** item basis. Command: `python scripts/run_model.py <name>`. The work-level
rows that superseded these as the published table are L52–L57; these stay because the
reasoning that produced the model choice happened here, and because every "read out loud"
paragraph below is still the argument, only re-based.

| ID | Model | HitRate@10 | Coverage@10 | Novelty@10 | Parameters | Measured |
|---|---|---|---|---|---|---|
| L22 | **popularity** (baseline) | **0.0145** | **0.019%** | 10.81 | rank by train interaction count, exclude own train items; `candidate_pool=2000` | 2026-08-04 |
| L24 | **item-item CF** (shrunk cosine) | **0.0546** | **9.064%** | 14.91 | binarized all-interaction matrix, shrinkage λ=10, 50 neighbours/item, min_support=1, score = Σ similarities over the user's train items | 2026-08-04 |
| →L44 | **item-item CF, work level** (M11) | **0.0644** | 8.190%* | 14.17 | identical model and parameters, fitted on the **work-keyed** matrix: 235,824 items, 13,580 eligible users. *\*Coverage is over 235,824 works, not 271,360 ISBNs — this row is **not** cell-comparable with L24.* **Not a line of its own: this is L44 restated in this table**, so the ISBN-level record carries its own successor row. It was mistakenly given the ID `L44` a second time; corrected 2026-08-09, and the ID belongs to the clustering line below. Superseded as the published row by **L53** | 2026-08-08 |
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
factors give item-to-item neighbourhoods that hold up where its HitRate does not (see L34,
and **L80–L82 for the measured version**: level with item-item on the item query, ahead of
everything on thin anchors, and below the baseline without its support floor — *this sentence
read "the single best item-to-item neighbourhoods of any model" until M20 measured it*); the
same fit yields user factors, so personalization is free the day the product has user identity;
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
| L34 | **ALS item-similarity needs a support floor, and with one it gives the best neighbourhoods of any model here** | 196,054 of 338,496 train items were touched exactly once; their factors are noise directions with mean norm 0.07 against 1.35 for items with 50+ interactions. With ~196k of them, the best chance alignment in 128 dimensions reaches cosine 0.95. **Unfiltered**, *Harry Potter*'s nearest neighbours were five one-reader books tied at 0.941. **With a floor of 20**: *The Fellowship of the Ring*, then Harry Potter 3, 2 and 4. *The Da Vinci Code* → *Angels & Demons*, *Digital Fortress*, *Deception Point* — all Dan Brown | Same factors, same formula: the noise simply outnumbered the signal at the argmax. Fixed by requiring 20 train interactions on the *similarity* endpoint only; `recommend` and every metric above are untouched. This solves L29's failure mode, on the model L33 says is otherwise the weakest — the ladder's rungs are good at different things. **Half of this row is superseded, 2026-08-09.** The *floor* half is confirmed and enlarged by **L82** — on 13,580 anchors it is worth +0.0177, the largest effect in the case. The *"best neighbourhoods of any model here"* half was three anchors read by eye, and **L80 retires it**: after the M12 re-base item-item returns the same Harry Potter list, and the item query cannot separate the two (p = 0.47). The claim was quoted in six documents for four milestones without being re-read against the re-base that overtook it; the wording is corrected in all of them and the mechanism is recorded with M20 | 2026-08-04 |
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
mechanism is the one the analysis predicted: co-occurrence counts that were split
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

Raised on 08.08 from the §8 gallery: the trailing parenthetical is not always a
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

## The work-level re-base (milestone M12)

L31 named the defect, L44 and L45 measured it on one model at a time, and M12 acts on it:
**the published comparison table is now keyed by work.** The ISBN-level table is kept above
as the journey record rather than deleted — it is how the model choice was reached, and the
distance between the two tables is itself the finding (L58).

The item universe is the M11 key including the transliteration extension (L40, L41, priced
at 1 wrong merge in 20 by L42). Every model runs unchanged: the content models read
`catalog.books` by its id column, and `work_level_catalog` hands them one row per work
carrying the title and author of that work's **most-interacted edition**, counted on train
only. Source for every line below: `python scripts/run_model.py --all --gallery
--work-level`, the two `--work-level` tuning scripts, and
`python scripts/decompose_work_level_lift.py`.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L49 | **The split, re-pinned on works** | **13,580 eligible users**, 1,129,755 train interactions, train matrix 105,283 × 303,381 | Identical mechanics to L19 — per-user leave-one-out, **seed 42**, eligible = ≥5 explicit ratings and ≥1 rating ≥8 — applied *after* `to_work_level` collapses every (user, work) pair, so holding out a work removes all of its editions from that user's profile at once. One of L19's 13,581 users loses eligibility when their graded editions merge. The train matrix has 303,381 columns rather than 235,824 because an interaction whose ISBN has no catalogue row becomes its own single-ISBN work — the same structure as L19's 338,496 columns over a 271,360-book catalogue | 2026-08-08 |
| L50 | **Merging editions barely moves the ceilings — the hybrid argument is invariant** | collaborative **86.66%** (L20: 84.81%) · content **88.98%** (L21: 89.31%) · union **95.34%** (L21: 95.37%) | `benchmark.ceilings`, the same three shares as L20/L21 recomputed on the work universe, and printed by *every* run so no table can quote a ceiling from the other basis. The collaborative ceiling rises 1.85 points because a work that was unreachable as a lone edition becomes reachable once its editions merge; content and union are flat to within 0.4 points. **The 8.7-point gap between collaborative-only and the union — the whole coverage argument for a hybrid — survives the re-base intact**, which is not something one could assume without measuring it twice | 2026-08-08 |
| L51 | *Null result, recorded because it is a result:* **both hyperparameter sweeps re-selected the ISBN-level values at work level** | item-item **λ=10, 50 neighbours** (0.0573 on the inner split); ALS **128 factors, α=1, reg 0.05** (0.0567) | `scripts/tune_item_item.py --work-level` (15 cells) and `scripts/tune_als.py --work-level` (6 cells), both on a leave-one-out split carved out of *train* (seed 43, **11,015 inner-eligible users**), never on the evaluation holdout. Item-item: λ=0 → 0.0379 at 17.049% coverage, λ=10 → **0.0573 / 6.842%**, λ=20 → 0.0551, λ=50 → 0.0514, λ=100 → 0.0464; 50 neighbours beat 200 and 500 at every λ. ALS at 128 factors: α=1 → **0.0567**, α=5 → 0.0518, α=20 → 0.0411; at 64 factors the same ordering, 0.0500 / 0.0438 / 0.0331. **This retires a caveat**: L44 called its +18% a lower bound because it ran on ISBN-tuned parameters. It was not a lower bound for that reason — the parameters were already the right ones. `models.WORK_LEVEL_PARAMS` is empty by measurement, not by omission | 2026-08-08 |
| L52 | **popularity** (baseline), work level | **0.0155** · **0.027%** · 10.54 | rank by train interaction count, exclude the user's own train works; `candidate_pool=2000`. **64 distinct works** recommended across all 13,580 users, against 51 ISBNs at ISBN level. 1.79% of the L50 ceiling | 2026-08-08 |
| L53 | **item-item CF** (shrunk cosine), work level — *the primary row* | **0.0644** · **8.190%** · 14.17 | binarized all-interaction matrix over works, λ=10, 50 neighbours/item, min_support=1, score = Σ similarities over the user's train works. Fit 22s, evaluation 2s. **Reproduces L44 to the digit** on an independently re-tuned parameter set (L51), which is the check that the re-base is deterministic. 7.43% of the L50 ceiling | 2026-08-08 |
| L54 | item-item, **explicit-only ablation**, work level | 0.0486 · 10.036% · 15.97 | identical model and parameters, fitted on the graded interactions alone over the same work index space. Discarding the ungraded rows now costs **24%** of the hit rate, against 31% at ISBN level (L26) — the same direction, a smaller penalty, because merging editions recovers part of what the explicit-only matrix was losing to fragmentation | 2026-08-08 |
| L55 | **ALS / weighted MF**, work level | 0.0545 · 0.897% · 12.29 | `implicit` ALS, 128 factors, α=1, regularization 0.05, 20 iterations, seed 42, similarity support floor 20 (L34). Fit 90s, evaluation 36s. Still loses to item-item on all three metrics and is still the most popularity-concentrated real model in the table — the L33 verdict is unchanged by the re-base | 2026-08-08 |
| L56 | **content TF-IDF** (coverage layer), work level | 0.0405 · **16.806%** · 17.07 | char_wb 3–5-grams over the canonical title+author of each work, min_df=3, **235,824 works vectorized, 215,377 features**. Fit 10s, evaluation 266s. This is the row the M12.6 plausibility gate stopped on: **+77.4%** against L30. Taken apart in **L58** | 2026-08-08 |
| L57 | **content embeddings** (multilingual), work level | 0.0141 · **26.143%** · **18.34** | `paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, all 235,824 works encoded from canonical title+author, profile vectors centered, score = mean cosine. Vectors cached separately from the ISBN-level set under `artifacts/embeddings/` — the cache key is a fingerprint of the text encoded, so the two sets coexist instead of overwriting each other. Still the coverage extreme. **On accuracy it is no longer distinguishable from the baseline**: 0.0141 against 0.0155 is 19 users of 13,580, z ≈ 1.0, and the paired test confirms it (190 wins to 210 losses, p = 0.342, L74) (reworded 2026-08-09 — the row previously read "still below the baseline on accuracy, now by 9% rather than 25%", which claimed a difference the sample cannot support). At ISBN level (L35) the same comparison was 0.0109 against 0.0145, z ≈ 2.7, and *did* clear the bar; the re-base moved embeddings up into the noise band, it did not move the baseline | 2026-08-08 |

**L53 read out loud.** Item-item beats the baseline **4.2× on accuracy and 302× on
coverage** at once, and captures 7.43% of the achievable ceiling against the baseline's
1.79%. Both ratios are *better* than the ISBN-level pair (3.8× / 477× — the coverage ratio
falls because the baseline's own reach grew from 51 items to 64 when its top list stopped
being split across editions). The model, its parameters and its rank in the table are
untouched; what changed is the data it was given.

**The item-to-item surface improved in a way no metric in this table can see, and it is
the most demoable result of the milestone.** L29 recorded that item-item answered *Harry
Potter and the Sorcerer's Stone* with two unrelated obscure books sharing four readers,
scoring 0.116, ahead of *Chamber of Secrets* at 0.097. On the work basis the same model,
same λ, answers: **Chamber of Secrets (0.477), Prisoner of Azkaban (0.424), Goblet of Fire
(0.380), Order of the Phoenix (0.271)** — the four sequels, in order, followed by
*Fellowship of the Ring*. The mechanism is the one L43 predicted: the anchor's evidence was
spread over 120 Harry Potter rows, and the shrinkage term was correctly refusing to trust
any single fragment of it. **This closes most of L29 as a data-prep consequence rather than
a re-tuning one** — the endpoint-specific λ that L29 proposed was never needed.

### Why the lift was not the same for every model — the decomposition

The M12.6 plausibility gate (band −20%…+40% relative to each ISBN-level row) **failed on
TF-IDF at +77.4%** and held the branch. No bug was found: leakage is zero at both levels,
and the held-out work cannot hide behind a sibling edition because `to_work_level` collapses
each (user, work) pair before the split. So the number was taken apart instead of waved
through. `python scripts/decompose_work_level_lift.py` re-runs **both** bases for all six
models — its work-level column reproduces L52–L57 to the digit, which is the check that it
is measuring the same thing the table does — and splits each lift in two:

1. **Evaluation fairness.** At ISBN level, recommending the Penguin edition when the reader's
   held-out book was the Vintage edition scores **zero**: the model named the right book and
   the metric called it wrong. Isolated with no re-fit at all — same model, same top-10 lists,
   only the definition of a hit changes (`eval.hit_rate_at_k_by_group`).
2. **Merged signal.** The residual: co-occurrence counts split across editions become one
   count, one canonical text per work replaces a pile of near-duplicate strings, the item
   universe shrinks, and a slot spent on a second edition of a book the reader already has
   goes to a real candidate instead.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L58 | **The +77.4% on TF-IDF is two effects, and the metric is the *smaller* one** | TF-IDF **0.0228 → 0.0250 (work credit) → 0.0405 (work basis)** = evaluation fairness **+9.4%**, merged signal **+62.3%**. Across all six models the fairness component is **+5.4% to +11.4%**; the merged-signal component runs **+0.5% to +62.3%** | Both bases re-run per model; the fairness column re-scores the stored ISBN-level top-10s under work credit. A fourth column repeats the fairness measurement with any slot the reader **already owns** blanked out (`serving.blank_owned_works`), because work credit could otherwise award a hit for recommending a third edition of a book the reader demonstrably has — something the work-level table can never do, since an owned work is blocked from the candidate list. That correction is negligible everywhere: TF-IDF 0.0250 → 0.0245, item-item 0.0588 → 0.0586 | 2026-08-08 |

All percentages in the table below are computed from the **unrounded** HitRates, so they will
not reproduce exactly from the four-decimal cells beside them (0.0405 / 0.0228 reads as
+77.6%; the run's own value is +77.4%, and that is the figure L56 and this section quote).
The last column is `duplicate_slot_rate` on the ISBN basis: **L45 tabulates four of these
six** from `analyze_dedup.py`; popularity and explicit-only were never in that four-model run
and come from `decompose_work_level_lift.py`, which computes all six.

| model | ISBN basis | + work credit | …owned blocked | work basis | evaluation fairness | merged signal | total | ISBN dup slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0145 | 0.0155 | 0.0155 | 0.0155 | +6.6% | +0.5% | +7.1% | 0.3% |
| item-item CF | 0.0546 | 0.0588 | 0.0586 | 0.0644 | +7.5% | +9.5% | +17.8% | 1.2% |
| ALS / weighted MF | 0.0451 | 0.0503 | 0.0501 | 0.0545 | +11.4% | +8.4% | +20.7% | 1.9% |
| item-item, explicit-only | 0.0379 | 0.0408 | 0.0406 | 0.0486 | +7.6% | +19.1% | +28.2% | 3.9% |
| content embeddings | 0.0109 | 0.0115 | 0.0112 | 0.0141 | +5.4% | +22.4% | +29.1% | 11.3% |
| **content TF-IDF** | **0.0228** | **0.0250** | **0.0245** | **0.0405** | **+9.4%** | **+62.3%** | **+77.4%** | **39.1%** |

**L58 read out loud — and it corrects the branch's own first explanation.** The gate note
argued that the lift was monotone in the duplicate-slot rate because the re-base "removes a
defect that was suppressing the text models specifically". The *total* column is indeed
monotone in that rate, in exact order across all six models. But the decomposition shows the
two halves behave completely differently, and only one of them is text-specific:

- **Evaluation fairness is roughly uniform and small — +5.4% to +11.4% — and it is not
  ordered by anything.** ALS has the *largest* fairness component (+11.4%) on a 1.9%
  duplicate rate; the embedding model has the *smallest* (+5.4%) on 11.3%. The ISBN-keyed
  metric was mildly unfair to **everybody**, which is a statement about the item key, not
  about text models.
- **The merged-signal component is what varies, by two orders of magnitude** (+0.5% for
  popularity, +62.3% for TF-IDF), and it is what tracks the duplicate-slot rate.

So the mechanism, named precisely: **the ISBN key charged the text models twice — once on
the output side, where 39.1% of TF-IDF's slots went to an edition the reader already had
(L45), and once on the scoring side, where naming the right book under the wrong ISBN scored
zero. Only the first charge was text-specific, and it is the one carrying the number.** The
second charge fell on everyone equally. L45 sizes the output-side path independently:
serving-time dedup alone, with nothing else changed, was worth **+21%** on TF-IDF
(0.0228 → 0.0277). That is a separate measurement rather than a sub-total of the +62.3% —
it is taken under ISBN credit and refills the freed slots — so it says the path is large,
not exactly how large a share of the residual it is. The rest of the residual is the
merged text and the merged profile: one canonical string per work instead of up to 53.

**What the gate got right, and what the band got wrong.** The gate was right to stop: a
number this far outside the band deserved exactly this examination, and the first explanation
offered for it turned out to be half wrong. The band was wrong because it assumed the re-base
was a re-parameterisation; for a model whose similarity is textual — and textually a reprint
and its original are the *same document* — it is also a defect fix. Both halves of that
sentence belong in the record.

**What this changes for the recommendation: nothing about the ranking, something about the
reading.** item-item > ALS > explicit-only > TF-IDF > popularity > embeddings on accuracy at
both item levels. But TF-IDF was **under-rated by the ISBN-keyed table by a factor, not a
rounding error**, so anyone reading L30 alone would overstate how far behind the content
layer sits. The hybrid argument gets stronger, not weaker.

### The gallery on the work basis: what the re-base fixed, and what it did not

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L59 | **The re-base cleans the collaborative surfaces completely and the content surfaces only halfway — the same shape L47 found, on a better basis** | Same-work neighbours surviving, out of 30 gallery slots per model: item-item **0/30**, item-item explicit-only **0/30**, ALS **0/30**, TF-IDF **7/30**, embeddings **6/30** | Hand count over the work-level 3-anchor gallery (3 anchors × top-10), the same rule as L47: a slot counts as a survivor only if it is the **anchor's own text under a different title** — a translation (*Desde Mi Cielo*, *In meinem Himmel*, *Harry Potter E la Pietra Filosfale*, *à l'école des sorciers*), an alternate regional title (*Philosopher's Stone*, the Welsh *Harri Potter maen yr Athronydd*), a subtitle variant (*The Lovely Bones* against the anchor's *The Lovely Bones: A Novel*), a dual-language title (*El Codigo Da Vinci / The Da Vinci Code*), or a re-credit to the illustrator (*Mary Grandpre* — which also carries a double space, so string normalization misses it twice over). Sequels, adaptations (the pop-up book, the movie poster book) and books *about* the anchor do not count. Worst single case: TF-IDF and embeddings on *Harry Potter*, 3/10 and 4/10 | 2026-08-08 |

### Complementarity, re-measured on the work basis

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L60 | **The two model classes reach mostly *different* books — the hybrid argument as an overlap rather than as two coverage percentages** | item-item **19,313 works (8.19%)**, TF-IDF **39,632 (16.81%)**, overlap only **6,794**, union **52,151 (22.11%)**; adding the embedding model takes the union to **93,992 works (39.86%)** | Distinct recommended items present in the work catalogue, over 235,824, taken from the same run that produced L53/L56/L57 — `notebooks/02_models.ipynb` §3. Same counting basis as L46 (a recommendation we cannot name does not count), now on the work universe. The ISBN-level analogue was 24,597 / 45,090, overlap 7,451, union 62,236 (22.9%) | 2026-08-08 |

**L60 read out loud.** The overlap is the number that carries the hybrid argument, and it is
small: of the 52,151 works the two classes reach between them, only **13% are reached by
both**. Two models with 8.2% and 16.8% coverage could in principle be nested; measured, they
are nearly disjoint. That is why the recommendation is item-item *with* a content layer
rather than item-item *or* a content layer — and it is a measurement rather than an appeal
to the idea that hybrids sound thorough. The union percentage barely moves from the ISBN
basis (22.11% vs 22.9%), so this argument, like the ceilings in L50, is invariant to the
re-base.

**L59 read out loud.** Compare with L47, which measured the same thing after serving-time
dedup on the ISBN basis: item-item 0/30, ALS 0/30, TF-IDF 7/30, embeddings 9/30. **The
collaborative surfaces were already clean and stayed clean; TF-IDF did not move at all;
embeddings improved by three slots.** Doing the merge in data prep rather than at serving
buys a great deal in the *metrics* (L58) and almost nothing on this particular surface,
because both approaches use the same title-equality key and therefore hit the same wall.

That wall is the honest limit of everything M11 and M12 did: *Harry Potter and the
Philosopher's Stone*, *Harry Potter à l'école des sorciers* and *Harry Potter E la Pietra
Filosfale* are one book with three names, and no amount of string normalization discovers
that from title+author. L38 found the same wall from the lookup side. **Two independent
measurements, one fix**: more text per book (LLM-generated descriptions, themes, genre
tags) or an external work identifier that already knows these are one book. That is a Part
3 proposal with evidence behind it rather than an opinion about LLMs.

## The demo app (milestone M13)

`streamlit run app/main.py` — paste a book, get ten similar books, each with one sentence
of grounded reason. Two things about it are **not** measurements and must not be read as
any: it is fitted on the **full** interaction matrix (serving, not evaluation — withholding
a reader's history would make the product worse for no reason), and it computes no metric.
The lines below measure the app *as an app*: does it start, does it answer, does the input
box find the book.

The engine is **ALS item factors over the work-keyed matrix** with the L34 support floor.
ALS is second of six on HitRate@10 (L55); the app asks a different question, and the sidebar
shows the table where ALS loses. **That second question is measured in M20** (L80–L83), and
the short version is that it does not name a winner: on the item query ALS and item-item are
not distinguishable, and in the band this app serves item-item is nominally ahead.

*Corrected 2026-08-09: this paragraph read "third of six" and so did the app's sidebar, one
line above a table that showed ALS second. The ordinal was read off the **ISBN-level** table
in M13, where the work-level item-item row (0.0644) sits beside the ISBN-level one (0.0546)
as a pointer, putting two item-item rows above ALS's 0.0451. Against the published work-level
table (L52–L57) there is one item-item row and ALS is second. **No measured value changes**;
the same wording is corrected in `app/main.py`, `recommender.demo`, `docs/model_selection.md`
and the milestone notes.*

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L61 | **The demo starts in 9 s and answers in 21 ms** | cold start **9.4s** (Streamlit ready 1.3s + first answer in a fresh interpreter 8.2s); warm query **21 ms** (lookup 21 ms + neighbourhood <1 ms), median of 15; assets **894 MB**, build **223 s** | `python scripts/measure_app_latency.py`. Cold start is measured in a *new* Python process — timing it in one that has already imported torch would measure nothing. 7.5 s of the 8.2 s is loading the sentence encoder for the free-text box; the assets themselves memory-map in 0.3 s. Against the Arbeitsplan DoD of cold start < 30 s and query < 1 s, with 3× and 47× of margin | 2026-08-08 |
| L62 | **The app's two lookup rules take the free-text box from 3/9 to 9/9, and neither touches a model** | resolved at rank 1: cosine alone **3/9** → + support floor **7/9** → + tie margin **9/9**. The two queries L38 identified as genuine failures (`"herr der ringe"`, `"hobit tolkien"`) fail under **all three** | `python scripts/audit_app_lookup.py`, on the L37/L38 query set plus three controls; "resolved" = the rank-1 title contains the expected work. **Rule 1, the support floor:** restrict candidates to works the engine can answer for, the same L34 floor. It removes the one- and two-reader books that were winning the argmax by chance — *Hoopla — Harry Stein* beating Harry Potter, exactly as L38 recorded. **Rule 2, a tie margin of 0.06 cosine:** among works within that margin of the best match, prefer the one with more readers. A margin rather than an additive popularity weight, because an additive weight can promote a *worse* text match when the readership ratio is large enough, and a margin cannot by construction | 2026-08-08 |

**L62 read out loud, including what it does not claim.** Rule 1 is a serving rule with an
argument behind it: offering an anchor whose neighbourhood the model would refuse to produce
is a dead end dressed up as a result. Rule 2 is a **user-interface** decision — a reader
typing a title means the book most readers mean — chosen on **nine queries**, which is a
small sample and is recorded as one. Neither rule exists anywhere near a published number:
the recommendation ranking never sees either, and no row in any table above changes by so
much as a digit.

The honest part is the last column of the audit. The floor and the margin fix the queries
that were failing for a *countable* reason — a book with one reader outranking a book with
832 — and they leave the two queries that fail for the reason L38 actually identified
exactly where they were. Title+author is three to five words, and no serving rule turns that
into enough signal for a multilingual encoder to bridge German to English. **That is the
same wall as L47 and L59**, hit from the input side, and it is the third independent
measurement pointing at the metadata-enrichment layer.

## Reading the demo's output (milestone M14)

M14 exists because someone read the app's answers for eleven anchors instead of reading a
table. Everything below was invisible to every cell in every table above. **The same rule
as M13 applies to all of it**: these lines measure the app, which is fitted on the full
matrix, so none of them is a model result and none may be quoted as one.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L63 | **The similarity score is not calibrated across anchors: it is highest where it is least trustworthy** | anchor support 20–30 → median **3.0** co-readers behind a shown book, **76.2%** of slots under 5, median similarity **0.507**; support 600+ → median **27.0** co-readers, **2.6%** thin, median similarity **0.354**. Evidence ×9, score **−30% in the wrong direction** | `python scripts/analyze_anchor_support.py` §1, 60 random nameable anchors per band, **seed 42**, top-10 each, statistics over all (anchor, slot) pairs. The mechanism: a factor fitted from 25 interactions is underdetermined and lands in a sparse region of the 128-dimensional space where high cosines are cheap; a factor pulled by 700 readers sits in a crowded region where nothing reaches 0.8. **This is L34's argument about candidates, applied to the anchor.** The top band holds only 19 nameable works, so its row is thin and says so. **Measured on the shipped assets** — see the re-measurement note below | 2026-08-08 |
| L64 | **The work key does not normalise whitespace before punctuation, and 1,198 works are one book counted twice** | 235,824 → **234,626** works (−0.508%); 1,198 merge groups over **2,580 ISBNs** (0.951%); **23,434 interactions** (2.038%) sit on a merged work but only **210 (user, work) rows** collapse (0.0184%); **47** groups have both sides above the support floor, i.e. are visible in the demo. Audit: **0 wrong merges in 30**. Priced on the published table: item-item HitRate@10 **0.0644 → 0.0649** (+0.8%), 13,580 eligible users unchanged, ceilings 86.66 → 86.70 | `python scripts/analyze_work_key_punctuation.py --write-sample docs/work_key_punctuation_sample.md`, seed 42; re-base priced with `python scripts/run_model.py item-item --work-level --work-key-punctuation`. 99.4% of the merges are the colon (`bridget jones : the edge of reason` = `bridget jones: the edge of reason`); the audit is clean because both sides carry *literally the same title*. **Decision: serving only.** The app builds on the fixed key, the published M12 table keeps the M11 key, and `artifacts/app/meta.json` records which is which | 2026-08-08 |
| L65 | **The floor was one number and had to be two: raising it for candidates buys evidence and pays in relevance** | anchor floor 20 → 50 → 100 → 200: askable works **7,541 → 2,508 → 959 → 339** (3.2% → 0.1% of the nameable catalogue), interaction coverage **39.7% → 26.6% → 17.3% → 9.9%**, thin slots **49.8% → 18.2% → 8.8% → 4.7%**. Raising *both* floors together instead takes thin slots to 9.4% at 50 — and costs *Dune* its **Heretics of Dune**, *Harry Potter* its **Quidditch Through the Ages** and *Fight Club* its **Trainspotting**: 35 of 110 slots across the eleven anchors change | `python scripts/analyze_anchor_support.py` §2/§3, same seed and sample. Interaction coverage is the honest denominator: the share of all interactions pointing at a work the engine would still speak about, i.e. how often a real reader's book can be answered at all. **Decision: anchor floor 50, candidate floor stays at 20 (L34).** Verified: every surviving anchor's top-10 is bit-identical to before, and `"da vinci code"` now resolves to the English edition instead of *El Codigo Da Vinci* (31 readers) | 2026-08-08 |
| L66 | **It is not returning bestsellers, and the contrast is what makes that a measurement** | over 300 random askable anchors × 10 slots: **2,149 distinct works**, the single most-recurring title appears in **2.7%** of lists, the 100 commonest works take **12.9%** of slots, and **72.8%** of recommended works appear in exactly one list. Pure popularity on the same anchors and the same candidate pool: **11 distinct works, every one in 100% of lists** | `python scripts/analyze_recurrence.py`, seed 42. Popularity computed as a sort of the same pool by `item_support` rather than by fitting `models/popularity.py`, because the app is fitted on the full matrix and the baseline on `split.train` — re-using the fitted baseline would mix two universes. **What it does not claim:** 64.1% of the demo's slots are in the global top 1% of works, but the candidate pool *is* already the top 3.2% of the catalogue, so that figure is a soft bar and is reported with its caveat rather than as a headline | 2026-08-08 |
| L67 | **The item-to-item surface is reproducible but not robust: a 0.5% change in the item universe replaces a third of every neighbourhood** | ALS item factors **bit-identical** across two fits on the same matrix (seed 42). Across the L64 re-key — 0.5% of works — the eleven anchors keep a mean of **6.8 of 10** neighbours: *Dune* and *Harry Potter* 9/10, but *To Kill a Mockingbird* **4/10**, *The Lovely Bones* 5/10, *Girl with a Pearl Earring* 5/10, *Bridget Jones's Diary* 5/10 | `python scripts/analyze_surface_stability.py`, comparison by title because the ids differ between the keys by construction. **Found by accident** — two anchors' answers changed visibly after a fix that touches 0.5% of the data, and the first suspicion (a non-deterministic optimizer) was checked first and ruled out. Reproducible and robust are different claims and only the first had ever been checked | 2026-08-08 |
| L68 | **Ten slots were a UI choice; only one of three candidate rules is a truncation at all** | relative score `score_i ≥ 0.55 · score_1` removes **1.4%** of slots over 300 random anchors and 14 of the eleven anchors' 110. The largest-gap (elbow) rule removes **76.4%**, median list **2**, 87% of lists cut below 5. A co-reader-share rule cannot truncate: mean Spearman between score rank and evidence rank inside a top-10 is **0.39**, and in **18.3%** of lists some slot carries ≥2× the evidence of everything above it | `python scripts/analyze_truncation.py`, seed 42. Relative rather than absolute because of L63 — an absolute cutoff would gut a well-supported list and leave a thin one whole. **Wired in at τ = 0.55 and then reverted on seeing it run**: it ended *Harry Potter* after the four sequels and *Bridget Jones's Diary* after four, and a list of four reads as a broken app in a live demo. `SCORE_TRUNCATION_TAU` is 0.0; the rule stays reachable per call. It would not have rescued *Guns, Germs, and Steel* either — every slot within 76% of its top score, every slot thin — because that anchor is the **floor's** problem, and conflating the two would be wrong | 2026-08-08 |
| L69 | **The M15 surface rebuild costs nothing at runtime and moves no number** | cold start **10.6 s** (against L61's 9.4 s; 9.5 s of it is the sentence encoder, which M15 does not touch), warm query **20 ms** (L61: 21 ms), assets **890 MB** unchanged. The "thin evidence" tag fires on **5.6%** of slots over the anchor sample and **27.3%** over the eleven demo anchors, both under the 30% ceiling above which the tag would be decoration. The evidence divider renders on **5 of the 11** anchors | `python scripts/measure_app_latency.py` and `scripts/analyze_anchor_support.py` §1b. The demo anchors are all well-supported and are therefore the tag's *worst* case, which is why their rate is five times the random sample's — the tag is a share of the anchor's readers, so it can fire more readily the more readers the anchor has. Nothing in M15 touches ranking, scoring, floors or truncation, and `pytest` pins that: no expected value in the M13/M14 suites changed | 2026-08-08 |

**Re-measurement note, and it is L67 happening to this ledger's own numbers.** L63 and L65
were first written from a run against the assets as they stood *before* the M14.4 work key
reached the serving path. Re-running the same script with the same seed against the
**shipped** assets moves them: the lowest band's thin share 72.0% → **76.2%**, the highest
band's median co-readers 35.0 → **27.0**, thin slots at floor 50 19.0% → **18.2%**. The
claims are untouched in direction, magnitude and sign; the digits moved because the item
universe moved by 0.5%, which is exactly what L67 measures. The shipped-asset numbers are
the ones above, because a ledger line that does not reproduce from what the app actually
serves is worth nothing. The earlier figures are kept here rather than deleted, since the
gap between them *is* a second instance of the finding.

**A rule the screen and this ledger now share (M15.4).** The app tags a suggestion as
**thin evidence** when fewer than **2% of the anchor's own readers** also read it — a UI
choice, labelled as one, like `LOOKUP_TIE_MARGIN`. `analyze_anchor_support.py` reports the
same rule as a column beside the absolute one, so the tag on screen and the measurement here
cannot drift apart. Self-check: it fires on **5.6% of 3,190 slots**, against a ceiling of 30%
above which it would be decoration. The two columns disagree by design and the disagreement
is the point: the absolute share falls with support (76% → 3%) while the share-based one
**rises** (0% → 36%), because at 20 readers 2% of the anchor is under one reader and nothing
can be tagged, while at 900 readers six shared readers is 0.7% and is. The tag is a statement
about *this anchor's* audience and can only be that. An absolute rule was the first proposal
and was dropped for the reason recorded in `display.py`: on the *Da Vinci Code* list "< 5
shared readers" fires on nothing, including the rank-2 row that started the discussion.

**L63 read out loud, because it is the line the write-up should carry.** A cosine of 0.49
against a 25-reader anchor and 0.49 against a 700-reader anchor are the same number meaning
different things. That is why the raw similarity no longer appears in the app's reason
sentences (M14.6) — showing it invites exactly the comparison it cannot support — and why
the truncation rule in L68 had to be relative. It is also the second time this project has
found the same shape of bug: L34 found it in the candidates, L63 finds it in the anchor.

**Two defects were fixed that no metric could see, and neither moves a published number.**
`same_author` was an exact string comparison, so *The Vampire Lestat* showed no author tag
under *Interview with the Vampire* while rank 2 did — the catalogue holds `ANNE RICE` beside
`Anne Rice`, `CHUCK PALAHNIUK` beside `Chuck Palahniuk`. And the work key kept
`bridget jones : the edge of reason` apart from `bridget jones: the edge of reason` (L64).
Both were visible on screen and invisible to every table.

**What M14 did not fix, measured rather than promised.** The remaining duplicates in the
demo are the *subtitle* class — *Lucky* beside *Lucky : A Memoir*, *The Hobbit* returning
another edition of itself as *The Hobbit: or There and Back Again*. Collapsing everything
after a colon would merge **9,523 further works across 30,486 clusters**, eight times L64's
reach and with real wrong-merge risk (*The Hobbit: or There and Back Again* should merge;
*Bridget Jones: The Edge of Reason* into *Bridget Jones's Diary* must not). That needs its
own sampled audit, not an evening — it is written up as an open item below.

Metric definitions, identical for every row (`src/recommender/eval.py`):
**HitRate@10** — share of eligible users whose held-out book is in their top-10. Under
leave-one-out this equals Recall@10, and Precision@10 = HitRate@10 / 10; one number,
three names. **Coverage@10** — distinct catalogue items appearing in any user's top-10,
over the whole catalogue; recommended ids absent from the catalogue do not count, because
a book we cannot name is a book we cannot show. **Novelty@10** — mean
`-log2((train_interactions + 1) / (total_train_interactions + catalogue size))`; the +1
smoothing keeps the metric finite for the zero-interaction books only content models can
reach.

**One denominator per table, and it is never mixed** (the mistake L46 records). The
work-level rows use **235,824**; the ISBN-level rows use **271,360**. Every run prints the
denominator it used together with the ceilings measured on the same universe, so a cell can
be traced to its basis without trusting this paragraph.

## The demo's surface, read critically (milestone M17)

M17 is display and serving only. **The ranking `demo.similar` returns is byte-identical
before and after** — demonstrated, not asserted: the eleven-anchor report was regenerated
and diffed against the run from before the branch, including every reason sentence. One line
below is a new measurement, because M17.4 adds a constant and this project does not ship a
constant without one.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L70 | **A lookup candidate more than 0.12 cosine below the best match is not an alternative reading of the query, and the cutoff cannot change what the query resolves to** | on-target alternatives sit a median **0.032** below the best match, off-target ones **0.212**. At 0.12 the cutoff keeps **79.5%** of on-target alternatives and **22.8%** of off-target ones — the widest separation the sample supports. The picker drops from 5 rows always to a median of **1** (56.7% one row, 16.3% five) | `python scripts/analyze_picker_margin.py`. 300 works above the anchor floor, each queried by **its own title**; a returned candidate is *on-target* when its title contains the query's or vice versa, and rank 1 is excluded because it is on-target by construction. The separation curve is a **plateau, not a spike** — 0.08 to 0.13 are within one point of each other — so the argmax was re-run on **six independent samples**: median **0.118**, range 0.102–0.153. 0.12 is that median at the resolution the sample supports; the third decimal is not claimed. It sits at **twice** `LOOKUP_TIE_MARGIN`, so it can never cut into the tie group and the resolved anchor is invariant, which is asserted in `tests/test_demo.py::TestPickerMargin` rather than left as a claim | 2026-08-09 |

**What L70 does not fix, and it is L38 again.** `"Guns Germs Steel"` keeps all five
candidates, because all five are Danielle Steel novels within 0.06 of each other. No cutoff
measured *relative to the best match* can see that, because the best match is already the
wrong book. `"little prince"` is the case it does fix: *A Little Princess*, a John Saul and
a Stephen King stop being offered beside the right answer.

**The lookup is reproducible against a fixed build (M17.9), so there is no line for it.**
Two screenshots of `"little prince"` had returned different candidate sets, and the
alternative to "the assets were rebuilt between them" was that `np.argpartition` selects
arbitrarily among equal scores — which would mean a rehearsed demo is not rehearsed. Run
against one unchanged build: five queries, three fresh interpreters, **byte-identical**
results, encoder vectors bit-equal, and **no exact score ties** anywhere in any shortlist.
The differing screenshots straddled the L64 re-key, and the two candidate sets differ in
exactly the way **L67** predicts a 0.5% change in the item universe will make them differ.
A green reproducibility check is a negative result and is recorded as one.

**An amendment to the L63 read-out-loud paragraph above.** It says the raw similarity "no
longer appears in the app". That is now true of the *reason sentences* and of the anchor
report, and false of the screen: since M17.1 the app shows the similarity as a bar plus a
number, because within one list the cosine **is** the sort key and the app only ever shows
one list. L63's incomparability is a claim *across* anchors, and the report — which prints
eleven anchors side by side — is exactly where it bites, which is why the two renderers
deliberately diverge (`demo.reason_sentence`, M17.7). The bar is scaled to the top of its
own list for the same reason: on a fixed 0–1 axis *The Little Prince* (0.33–0.40) would
render as a row of stubs beside *Interview with the Vampire* (0.50–0.80).

**Unasked-for finding, and it was taken: the double encoding is repaired for display.**
The source CSV holds *Antoine de Saint-Exupéry* as `Saint-ExupÃ©ry` — UTF-8 bytes written as
latin-1 characters, **a defect in the file, not in how this project reads it**. The repair is
the inverse of the original mistake (`encode("latin-1").decode("utf-8")`) in
`demo.repair_encoding`, called from `DemoEngine.describe` beside the `html.unescape` that
already fixes the same class of defect one layer up.

**Its own failure is the guard, so nothing is guessed.** A string that was never
double-encoded produces bytes that are not valid UTF-8 — a genuine *é* is the single byte
`0xE9` — so it raises and comes back untouched. Over all **469,252** title and author
strings in the shipped catalogue: **3,234 change, and every one carries the `Ã`/`Â`
signature — zero change without it.** One pass is a fixed point: no string needs a second
and no repaired string repairs again, which is why it is not a loop.

| where it lands | works | repaired |
|---|--:|--:|
| reachable as an **anchor** (floor 50) | 2,508 | **1** — *The Little Prince* |
| reachable as a **candidate** (floor 20) | 7,541 | **12** — incl. *Le Petit Prince*, two Spanish *Harry Potter* volumes, `John Le CarrÃ©` |

*(Corrected 2026-08-09: an earlier draft of this paragraph said 2 anchors. That count came
from a regex for the `Ã`/`Â` signature, which also flags strings the repair correctly leaves
alone; the number of works the repair actually changes is 1.)*

**No tag can flip, checked rather than assumed.** The one thing downstream that reads this
text is the `same_author` comparison, and it takes *both* sides from `describe`, so the two
are always cleaned to the same standard — the M14.3 lesson one layer down. The risk left is a
repaired string colliding with a *different* raw one: across all 7,541 candidates, **zero**
repaired author strings merge two distinct raw strings. The eleven anchors re-run
byte-identical, tags and reason sentences included.

**Display only, and that is load-bearing.** The work key is built upstream from the raw
column and still reads `the little prince|saintexupãry`. Repairing it there would move the
edition merge and with it published numbers, so it is deliberately left corrupt and pinned by
a test. Nothing here is a data-layer repair, and no ledger number moves.

## What the system costs to serve (Part 3)

Part 3 argues an architecture, and an architecture argued without sizes is a box diagram.
These two lines are the sizing evidence, measured against the **shipped** assets rather
than estimated at a whiteboard. Source: `python scripts/measure_serving_footprint.py`,
which reads only and fits nothing, and which prints the measured and the derived rows in
separate blocks because they are different kinds of claim. **Nothing here changes a model
result** — no model is fitted, scored or re-ranked to produce any of it.

The timings quoted below are not new measurements: they are the fit and evaluation times
already recorded in L53–L57 and L61/L69, gathered into one place so that "what does a
retrain cost" has an answer that traces line by line.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L71 | **The demo ships 890 MB and the recommender is 17.5% of it — the rest is the search box and a padded id column** | assets **890.0 MB**: free-text lookup (encoder vectors 234,626 × 384 float32, plus ids and support) **372.2 MB / 41.8%**, the item id column **328.3 MB / 36.9%**, the ALS factor matrix that actually answers the query **155.6 MB / 17.5%**, the reader matrix **3.0 MB**, catalogue parquet **28.4 MB**. The id column is stored as fixed-width `<U270`, i.e. 1,080 bytes per id whatever the id's length; the same ids as int32 codes plus a utf-8 dictionary are **11.4 MB, 96.5% smaller** | Byte sizes of every file in `artifacts/app/` as the demo loads them, plus shapes and dtypes; the dictionary figure sums the utf-8 length of all 304,001 ids. **This is a finding about the demo's asset builder, not about the method** — it is a serving-layer packing choice and no published number depends on it. It is the concrete answer to "what is the model artefact": 155.6 MB of float32, and it would be 155.6 MB on any platform | 2026-08-09 |
| L72 | **A precomputed answer table for the whole product is 1.5 MB — 591× smaller than what the demo ships** | **2,508 askable anchors** (support ≥ 50, the L65 floor) × top-10 = **25,080 rows / 0.30 MB** per engine, **125,400 rows / 1.5 MB** for the five-engine shortlist, at 12 bytes a row (int32 anchor, int32 item, float32 score). For contrast: item-item at 50 neighbours per item is 15,200,050 entries / **121.6 MB**, and the dense similarity matrix nobody ever builds is 92,416,608,001 cells / **370 GB** | Derived arithmetic on measured inputs (`n_items` = 304,001 and the anchor support vector from the shipped `meta.json` and `item_support.npy`; 50 neighbours from L53, 128 factors from L55). Every MB here is 10⁶ bytes, the same convention as L71. *(Corrected 2026-08-09 by M18.5's sweep: this line first said 2,532 anchors, which is `support ≥ 50` over all 304,001 item rows, **24 of them ids the app has no catalogue row for and therefore cannot name**. "Askable" is L65's word and L65's filter — an anchor a visitor cannot type is not askable — and the nameable count is 2,508, which is what L65's floor table already published and what the paragraph below always quoted at floor 20. The design conclusion is untouched: 1.5 MB either way.)* **This is the number that decides the serving design**: at 1.5 MB the answer table fits in any cache, a key-value store is sufficient and neither a vector database nor a live model server is required for the shipped use case. They become required exactly when the anchor floor comes down or personalization arrives, and that is the trade to state on the slide rather than the technology | 2026-08-09 |

**L71 and L72 read out loud, and it is one argument.** The demo is 890 MB because it carries
a sentence encoder's output so a human can type a title, and because ids were written out as
padded Unicode. **The recommender is 155.6 MB, and the answers it produces are 1.5 MB.**
Those three numbers in that order are the whole productionization story: the expensive part
is the lookup, not the model; the model is small; and the served artefact is smaller still,
because for this use case (anonymous reader, one book in, ten books out) every answer can be
computed in advance. That is why the architecture is a batch job writing a small table, and
not a model server — and it is a measurement, not a preference.

**The floor is what moves these numbers, and the honest caveat comes with it.** 2,508
anchors is **1.1% of the 234,626 works**: the system can answer for the books people
actually read and refuses the rest (L65 prices the refusal — at floor 50 the askable works
cover 26.6% of all interactions). Drop the floor to 20 and it is 7,541 anchors, the table is
still under 5 MB, and thin slots go from 18.2% to 49.8%. **The serving cost is not what
constrains the product; the evidence is.**

**What a retrain costs, gathered rather than newly measured.** Every cell below is already
in a line above; they are collected here because "how often would you retrain" is a Part 3
question and the answer should be an addition, not a shrug. Single node, Apple Silicon,
1.13M train interactions over 304k items.

| Step | Time | Line |
|---|---:|---|
| popularity fit | instant | L22 |
| item-item fit (50 neighbours, λ=10) | 22 s | L53 |
| ALS fit (128 factors, 20 iterations) | 90 s | L55 |
| TF-IDF fit (235,824 works vectorized) | 10 s | L56 |
| embedding encode of the whole catalogue | 151 s | L35 / L57 |
| **all six models, fit + evaluate + gallery** | **~8 min** | primary table header |
| build the demo's serving assets | 223 s | L61 |

**Read out loud: a full retrain of everything is eight minutes on a laptop.** That is the
number that decides the retraining cadence, and it decides it in an unusual direction —
compute is not the constraint, so the cadence should be set by how fast the *catalogue and
the interactions* change, not by what the training costs. **This dataset cannot tell us how
fast that is (L18: there are no timestamps)**, which makes it a discovery question for the
client rather than a parameter to assert on a slide. Daily is defensible and cheap;
justifying it from this data is not possible, and saying so is the stronger answer.

*Note on 2,508, and on the 2,532 this line first said.* The two numbers are the same filter
run with and without one clause, on the same shipped assets: `item_support.npy` has
**2,532** rows at support ≥ 50 and **2,508** of them have a catalogue row the app can name.
L65's askable count is the nameable one, so 2,508 is the number that belongs here, and the
shipped assets reproduce L65's whole floor table exactly — 7,541 / 2,508 / 959 / 339 at
floors 20 / 50 / 100 / 200. **This note previously attributed the 24-anchor gap to the M14.4
re-key (L67) and that was wrong**: the re-key explanation was written from the plausible
mechanism rather than from a re-run, and the re-run shows the gap is entirely the missing
nameable filter. Corrected by M18.5's sweep, 2026-08-09. It is a small number with a general
lesson attached, which is why the wrong version is kept visible rather than deleted: a
difference that has a plausible cause is exactly the kind that never gets measured.

## How certain is any of this, and where does it come from (milestone M18)

Two questions the ledger had never answered about its own primary table, both answered from
**one** run of the six models on the pinned work-level split (L49) and **one** artefact: the
per-user hit vector, 13,580 booleans per model, cached to `artifacts/significance/`. L73
groups those vectors; L74 compares them. Every published cell is asserted against this run
before either question is asked — `scripts/measure_significance.py` exits non-zero if one has
moved — and on 2026-08-09 **all eighteen reproduced**: six models × HitRate@10 to four
decimals, Coverage@10 to three decimals of a percent, Novelty@10 to two. That check is the
precondition for the milestone, not a by-product of it: an interval around a number that has
quietly drifted would be worse than no interval at all.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L73 | **L27 and L28 at work level: the baseline is still narrow, item-item still degrades on long profiles, and the content models are the only ones that score where no reader has been** | HitRate@10 by the **held-out work's train support** (0 / 1–4 / 5–49 / 50+ over 1,812 · 2,410 · 4,591 · 4,767 users): popularity **0.0000 / 0.0000 / 0.0000 / 0.0443**, item-item **0.0000 / 0.0170 / 0.0571 / 0.1198**, ALS 0.0000 / 0.0000 / 0.0109 / **0.1447**, item-item explicit-only 0.0000 / 0.0116 / 0.0431 / 0.0910, TF-IDF **0.0304 / 0.0303 / 0.0414 / 0.0487**, embeddings **0.0138 / 0.0091 / 0.0155 / 0.0153**. By the **reader's train-profile length** (0–9 / 10–24 / 25–74 / 75+ over 3,562 · 4,596 · 3,153 · 2,269 users): item-item **0.0679 / 0.0716 / 0.0695 / 0.0370**, ALS 0.0528 / 0.0568 / 0.0603 / 0.0445, TF-IDF 0.0528 / 0.0424 / 0.0390 / **0.0194**, embeddings 0.0225 / 0.0150 / 0.0105 / **0.0040**, popularity 0.0174 / 0.0159 / 0.0146 / 0.0132, explicit-only 0.0522 / 0.0472 / 0.0555 / 0.0361 | `python scripts/analyze_hit_strata.py`, grouping the cached hit vectors; support and profile length both counted on **train only**. The strata boundaries are L27's and L28's unchanged, so the columns line up with the ISBN-level lines. Cross-check that the split is the one it claims to be: the leftmost stratum is **1,812 users = 13.34%**, i.e. exactly L50's `100% − 86.66%` collaborative ceiling. **This closes the state the notebook was in** — `notebooks/02_models.ipynb` §2.1 printed the popularity row of the first table and nothing recorded it. The oracle bound these vectors also give is below the table | 2026-08-09 |
| L74 | **The paired test confirms the ranking and disarms exactly the one cell the derivation disarmed: embeddings against the baseline** | 95% Wilson intervals: item-item **0.0644 [0.0604, 0.0686]**, ALS 0.0545 [0.0508, 0.0584], explicit-only 0.0486 [0.0451, 0.0523], TF-IDF 0.0405 [0.0373, 0.0439], popularity 0.0155 [0.0136, 0.0178], embeddings 0.0141 [0.0122, 0.0162]. Paired McNemar **against item-item**, every model: ALS 337 wins / 471 losses, **p = 2.7e-06**; explicit-only 204/418, p = 6.3e-18; TF-IDF 288/612, p = 1.3e-27; popularity 161/824, p = 8.5e-108; embeddings 87/770, p = 1.8e-137 — **all five distinguishable**. Against the baseline: item-item p = 8.5e-108, ALS p = 6.8e-79, explicit-only p = 1.2e-57, TF-IDF p = 7.1e-36 — and **embeddings 190 wins / 210 losses, Δ = −0.0015 [−0.0044, +0.0014], p = 0.342: not distinguishable** | `python scripts/measure_significance.py`. Exact two-sided binomial on the discordant pairs; Wilson rather than Wald because these proportions are small. The paired interval on a difference is `(b−c)/n ± 1.96·√(b+c)/n`, tighter than the unpaired one because the models hit largely the same users. **The prediction under the primary table held**: the unpaired derivation put embeddings-vs-baseline at z ≈ 1.0 and the two cells were reworded to "not measurably different" on that basis on 09.08, *before* this ran; the paired test agrees (p = 0.342) and no wording had to be reverted. The narrowest comparison that still clears the bar is ALS against item-item — 0.0099 apart, p = 2.7e-06 — so the table's ordering is safe everywhere except the one pair already labelled | 2026-08-09 |
| L75 | **The consistency sweep: 479 numeric literals across every artefact a reader can open, six of them wrong** | **479** numeric literals in `README.md`, `docs/*.md`, both notebooks' markdown, `app/main.py` and every module docstring, each checked against this ledger — 479 is the count *before* the corrections, and a re-run after them scans **508**, because every correction quotes the number it replaces. **Six wrong**, all corrected and each carrying its old wording: L72's **2,532** askable anchors (nameable filter missing, 2,508), `demo.py`'s "**a series 27% of the time**" (L48's 27.1% is volume/part numbers), `analyze_recurrence.py`'s **7,523** works above the candidate floor (L65: 7,541), and three in `notebooks/02_models.ipynb`. Four further claims were **stale rather than wrong**, and two numbers were sound but unsourced — both classes are itemised below the table | Literals extracted mechanically (dates, DOIs, ISBNs and section numbers excluded), then every survivor adjudicated by hand against the line it should trace to, plus a second independent read for claims that quote a *right* number for a *wrong* thing — which is where four of the six came from, since a wrong number that exists elsewhere in the ledger passes a verbatim check. The two unsourced figures were re-measured on the shipped assets (`artifacts/app/books.parquet`, and `DemoEngine.similar` on the resolved anchor) and reproduced exactly | 2026-08-09 |

**L73 · the oracle bound, which is a bound and not a result.** TF-IDF and embeddings between
them hit **320** users item-item missed, **61** of those at zero support, so a chooser that
always picked the right one of the three would score **0.0879** against item-item's 0.0644
(item-item with TF-IDF alone: 0.0856). **No hybrid scores that** — picking the right model per
user is the whole problem, and M19 measures what a real rule gets.

**L75 · the three wrong numbers in the notebook, and what went with L72's correction.** In
`notebooks/02_models.ipynb`: "**below the popularity baseline**" (L74 says tie), "**ten points
at either item level**" (L50: 8.7 at work level, 10.6 at ISBN) and "reverse of the coverage
ranking **with ALS the only exception**" (the primary table's own correction). L72's fix also
removed the note that had explained its 24-anchor gap as the M14.4 re-key, which was not the
reason: the gap is the nameable filter.

**L75 · the four stale claims and the two unsourced numbers.** Stale rather than wrong: three
said intervals and work-level strata were unmeasured, which L73/L74 have since closed, plus one
pointer the sweep called dangling that was not — `model_selection.md` §9 item 6 existed and had
been lost to a concurrent edit, and it was restored the same evening. Sound but unsourced, and
now carried by this line: the `series` parenthetical holds *Penguin Classics* on **378** works
and *Dover Thrift Editions* on **268**; and the M17 evidence example — on *The Little Prince*,
rank 1 shows **16** shared readers and rank 2 **17**, at cosines 0.3982 and 0.3516.

**L75 · the class of error, and the sweep's own blind spot.** Nothing was miscalculated. Five
of the six are a number that was correct when written and was overtaken by a later measurement,
which is the failure mode a ledger is supposed to catch and only catches if something sweeps.
**The sweep then walked straight into its own blind spot the same evening**: it extracts
*numeric literals*, so "ALS places **third of six** on HitRate@10" in `app/main.py` and
`model_selection.md` was invisible to it — an ordinal written as a word, wrong since the M12
re-base (ALS is second), sitting on the demo's own sidebar above a table that said second.
Corrected 09.08.2026. A ranking claim is a number; a regex for digits is not a sweep for claims.

**What L74 changes about how this table should be read, and what it does not.** It does not
promote a single row: the ranking it confirms is the ranking that was already published. What
it removes is one specific over-claim — that content embeddings are *worse* than recommending
bestsellers — and it removes it with a test rather than an argument. **13,580 users can order
five of these six models and cannot order the sixth against the baseline**, and that is the
honest sentence. The intervals also set the resolution of every future comparison on this
split: **±0.002 to ±0.004**, so a hybrid, a re-tune or a new model that moves HitRate by less
than about 0.004 has not been shown to move it at all.

**And the uncertainty these intervals do *not* cover, because it is the larger one.** Both
L74's interval and its paired test hold the drawn holdout fixed and quantify sampling across
*users*. Seed 42 also decides **which** of a reader's favourites is held out, and L73 is the
argument that this second draw is not a detail: HitRate differs by a factor of seven between
the lowest and the highest support stratum, so which stratum a reader's drawn book lands in
matters more than most model differences in the table. Measuring it means re-running the
whole comparison on several seeds — **done, M21 (L86)**: five draws, and the ordering holds on
all of them, but ALS against item-item — the narrowest pair this table calls separable — is
separable on only **four of the five**, and the p = 2.7e-06 below is seed 42's draw. Read that
cell with L86 beside it. *(The estimate of eight minutes a seed was wrong by a factor of three:
it is 22, because the three M19 hybrid rules cost 14 of them.)*

*One property that makes those seeds comparable, recorded because it was briefly got wrong
in `model_selection.md` §9 and corrected on 09.08.* **Eligibility does not depend on the
seed.** In `split.py` it is an `intersect1d` over two data thresholds (≥5 explicit ratings,
≥1 rating ≥8); the seed enters at a single `rng.integers` call that draws the held-out item.
So a multi-seed run scores **the same 13,580 users** every time and the runs differ only in
which book each of them had withheld — which is precisely the variance to be measured, with
nothing else moving underneath it. Every
number in this ledger is therefore conditional on one draw, and that sentence belongs in the
talk track next to the interval, not instead of it.

## The hybrid, measured at last (milestone M19)

Since M10 this project's headline recommendation has been *item-item as the scoring core,
with the content layer serving the catalogue it structurally cannot reach*. That sentence
rested on two **bounds** — L50 (the union raises the achievable ceiling from 86.66% to
95.34%) and L60 (the two model classes reach different parts of the catalogue) — and never
on a row. M19 runs it: three combination rules over the *same* two fitted models, on the
pinned work-level split, with `python scripts/measure_hybrid.py --tune --gallery`.

**The prediction, written before the run and quoted here unedited** (09.08.2026):
*"Coverage will move a lot and HitRate will barely move, possibly down. The held-out items
only the content layer can reach are by definition items with little or no interaction
evidence… the 8.7 points of extra reachable ceiling are reachable in principle and mostly
unrankable in practice."*

**Verdict: falsified as stated, and held in its reasoning.** No rule produced the predicted
combination. The rule the recommendation actually describes (cascade) moved coverage
*barely* — 8.190% → 8.529%, not "a lot" — and moved HitRate by nine users, exactly as
predicted. The two fusion rules moved coverage a lot **and** moved HitRate measurably up,
which the prediction ruled out. The reasoning underneath it was right and is the more useful
half: the pure coverage play is worth **9 users out of 13,580**. Everything above that comes
from somewhere the prediction never considered — reranking readers the collaborative model
already serves — and L79 is why that is not free.

Both base models were re-fitted and checked before anything was combined: item-item
(0.0644 · 8.190% · 14.17) and TF-IDF (0.0405 · 16.806% · 17.07) reproduce to the digit and
their per-user hit vectors are **bit-identical** to M18's. The escalation gate held too:
score fusion beats item-item by **+7.1% relative**, inside the +15% band M19 set, so no
plausibility escalation was triggered.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L76 | **The hybrid this project has recommended for four milestones is worth nine users out of 13,580 — and it is the only rule that costs nothing** | **Cascade / backfill**, item-item filling the list and TF-IDF filling what is left: HitRate@10 **0.0644 → 0.0650** (883 hits against 874), Coverage@10 **8.190% → 8.529%**, Novelty@10 14.17 → 14.19, slots filled 99.585% → 99.887%. Paired McNemar against item-item: **9 wins, 0 losses, p = 0.0039** — distinguishable, and every one of the nine is a pure addition. By held-out support: **+8 at zero support, +1 at 1–4, nothing at all above that, and no losses in any stratum.** With a support floor of 5 it becomes 0.0652 but **+51/−39, p = 0.246 — not distinguishable**, and coverage *falls* to 6.955%; at floor 20 it is 0.0596, +104/−169, **p = 0.0001 distinguishably worse** | `python scripts/measure_hybrid.py`, work-level split L49, both base models untouched and re-checked first. The floor variants price the second half of the rule ("or where a slot would go to an item below a support floor"): raising it buys thin-stratum hits and pays for them in the 5–49 band at a rate that turns negative between 5 and 20. **Coverage falling as the floor rises is not a paradox** — the floor evicts exactly the low-support collaborative items that were the coverage | 2026-08-09 |
| L77 | **Score fusion is the accuracy winner and the tuning that produced it bought nothing** | `α · norm(item-item) + (1-α) · norm(TF-IDF)`, per-user min-max, **α = 0.6 chosen on the inner validation split**: HitRate@10 **0.0690** (937 hits, 95% CI [0.0649, 0.0734]), Coverage@10 **11.912%**, Novelty@10 14.78. Against item-item: **+175/−112, Δ = +0.0046 [+0.0022, +0.0071], p = 0.00024**. The whole α curve on validation (11,015 users, seed 43): 0.0378 · 0.0408 · 0.0441 · 0.0468 · 0.0539 · 0.0582 · **0.0617** · 0.0616 · 0.0610 · 0.0587 · 0.0578 for α = 0.0 … 1.0 — a broad plateau from 0.6 to 0.8, not a peak. **Against RRF, which has no parameters at all: 162/158, Δ = +0.0003, p = 0.867 — not distinguishable** | Same run, same candidate lists. α was tuned on `inner_bench(seed=43)`, the harness L51 used, and never on the holdout; the curve is reported because the argmax of a plateau is a summary and not a result. **M19 asked this question in advance and this is the answer it gets:** a tuned mixing weight cannot be told apart from a constant, so the honest description of fusion here is "combine the two rankings", not "combine them at 0.6" | 2026-08-09 |
| L78 | **Reciprocal rank fusion matches the tuned fusion on accuracy and beats every rule on reach** | `Σ 1/(60 + rank)`, no tuning: HitRate@10 **0.0687** (933 hits), Coverage@10 **12.843%** — the widest of any rule and **+57% relative** over item-item — Novelty@10 **15.14**. Against item-item: **+312/−253, Δ = +0.0043 [+0.0009, +0.0078], p = 0.015**. Only **43.2%** of its slots are slots item-item would have filled, against 96.4% for the cascade | Same run, k = 60 left at the TREC default on purpose: RRF is in this comparison as the parameter-free control, so tuning it would remove the reason it is here | 2026-08-09 |
| L79 | **The two rules that win the metric are the two that break the demo, and the mechanism is not metric leakage** | Three-anchor gallery, hand-counted the L47/L59 way — a slot counts as bad when it is the anchor itself under another edition, title or translation, a companion book about the anchor, or a duplicate of another slot: **cascade 0 of 30, fusion 11 of 30, RRF 13 of 30**. RRF answers *The Lovely Bones* with *The Lovely Bones* (rank 2), *Desde Mi Cielo* (7) and *In meinem Himmel* (9) — the same book three times. Measured over all users, share of slots whose loose title the reader already owns: cascade **0.52% of slots / 3.9% of users**, fusion **4.30% / 28.8%**, RRF **5.57% / 33.9%**. **But the metric is not being gamed:** of fusion's 175 new hits only **7** are a work whose loose title the reader already owned, and of RRF's 312 only **11**. What the fusion rules actually do is **redistribute**: by held-out support, fusion is +33/−0 at zero, +30/−4 at 1–4, +64/−33 at 5–49 and **+48/−75 at 50+**; RRF is **+137/−178** in that last stratum | Gallery from the same run (`--gallery`); the duplicate share uses a loose title key (drop everything from the first `:` or `(`, then non-alphanumerics) and is a **lower bound**, because it cannot see translations — *El Codigo Da Vinci* shares no characters with *The Da Vinci Code*, which is L47's standing ceiling. Pairwise tests via `--pairs-from`. **This is L58's finding in a new place**: the offline metric and the product surface disagree, and here they disagree about the same rule at the same time | 2026-08-09 |

**Read out loud, because this is the paragraph that goes on the slide.** The hybrid works,
and it does not work the way we have been saying it does. The version this ledger has
recommended since M10 — collaborative first, content filling what it cannot reach — is
**real, statistically distinguishable and tiny**: nine readers out of 13,580, every one of
them in the stratum where the collaborative model has nothing, and not one hit lost anywhere.
That is the coverage argument, measured, and it is worth about half a percent of item-item's
hits. The rules that win more — score fusion at +7.1% relative, RRF at +6.7% — do not get
there by reaching further into the catalogue. **They get there by taking accuracy from
well-evidenced readers and giving it to thin ones**, and by filling a third of readers' lists
with another edition of a book they already have. Neither of those is visible in HitRate.

**So what we would actually ship, now measured rather than bounded:** item-item as the
scoring core with the content layer as a **backfill only** (L76). It is the one rule that
adds without subtracting, it leaves the demo's answers bit-identical to the item-item gallery
L59 already vouched for, and its cost is one extra model at serving time. **Score fusion is
not recommended, and the reason is not its HitRate** — it is that 28.8% of readers would see
their own book recommended back to them, which no offline metric in this project penalises
and every reader would notice. If the edition/translation clustering L47 describes were
fixed, fusion becomes the candidate to re-measure first; the ordering of that work is now
evidence-backed rather than a hunch.

## The question the demo actually asks (milestone M20)

Every accuracy row above scores a **profile query**: given this reader's history, rank their
held-out book. The demo asks an **item query**: given this one book, what is like it. The
project has said those are two different questions since M13 — and then answered the second
one with **three anchors read by eye** (L34, 04.08.), *before* the work-level re-base that
made item-item's and ALS's *Harry Potter* neighbourhoods identical. Since M12 no published
number has separated them on the surface the app serves, and six documents went on calling
one of them "the best". This section is that claim replaced by a column.

**AnchorHitRate@10.** One anchor per eligible reader, drawn from that reader's *train* rows;
the model's top-10 neighbours of that anchor with the reader's own train items filtered out;
a hit when the held-out book is among them. Same 13,580 readers, same held-out books, same
row order, and `hit_vector` unchanged — so every row here pairs against its own row above.
Command: `python scripts/measure_anchor_hitrate.py`.

**The anchor rule is a free parameter and is therefore part of the measurement**, not a
detail of it: uniform seeded draw (seed 42) from the reader's train ratings ≥8 — the same
population `make_split` draws the holdout from — falling back to any graded train row for the
**949 readers (7.0%)** whose only ≥8 rating *was* the holdout.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L80 | **On the question the demo actually asks, the engine it runs and the model that wins the accuracy table cannot be told apart** | AnchorHitRate@10 over **13,580 anchors**, with 95% Wilson intervals: hybrid cascade **0.0317** [0.0289, 0.0348], **ALS 0.0308** [0.0280, 0.0338], **item-item 0.0297** [0.0270, 0.0327], TF-IDF 0.0258 [0.0233, 0.0287], item-item explicit-only 0.0208 [0.0186, 0.0234], popularity 0.0155 [0.0136, 0.0178], ALS without its support floor 0.0130 [0.0113, 0.0151], embeddings 0.0126 [0.0108, 0.0146]. Paired McNemar, **ALS against item-item: 169 wins / 155 losses, Δ = +0.0010 [−0.0016, +0.0036], p = 0.47 — not distinguishable.** Cascade against ALS 182/169, p = 0.52, also not distinguishable; cascade against item-item 29/2, p = 4.6e-06 | `python scripts/measure_anchor_hitrate.py`. Anchors from `recommender.split.pick_anchors`, scoring from `recommender.eval.evaluate_anchors`; the reader's train items are filtered out of every list, because `recommend` excludes them by contract and comparing a filtered list against an unfiltered one would test the filter rather than the query. A model returning nothing is a **padded row and a miss**, never a dropped reader — `answered` is 100% for the collaborative models and **89.6%** for both content models, the share of anchors with a catalogue row. **The control that validates the harness:** popularity's `similar_items` is the global top-10 for every anchor, so its two columns must be the same measurement — they are **0.0155 and 0.0155, 0 discordant readers, bit-identical vectors**. Seed sensitivity below the table | 2026-08-09 |
| L81 | **ALS's item-to-item advantage lies entirely in the band the app refuses to answer, and in the band it does serve item-item is ahead in both draws** | AnchorHitRate@10 by the **anchor's** train support (1–4 / 5–49 / 50+ over **3,946 / 4,829 / 4,805** anchors at seed 42): ALS **0.0162** / 0.0220 / 0.0516, item-item **0.0053** / 0.0246 / **0.0549**, cascade 0.0124 / 0.0244 / **0.0549**, TF-IDF 0.0155 / 0.0290 / 0.0312, embeddings 0.0091 / 0.0145 / 0.0135, popularity 0.0084 / 0.0114 / 0.0256. ALS against item-item per band, **both anchor draws**: **1–4 → 64/21 (p = 3.3e-06) and 79/27 (p = 4.3e-07), distinguishable in both**; 5–49 → 54/67 (p = 0.28) and — ; **50+ → 51/67 (p = 0.167) and 49/85 (p = 0.002)**, i.e. item-item ahead in both draws and **distinguishably ahead in one of the two**. The app's anchor floor is **50 readers** (L65), so the served band is **4,805 anchors, 35.4%** | Same vectors as L80, grouped — one definition of a hit, per `eval.py`. L73's boundaries minus its leftmost bin: an anchor *is* a train interaction, so support 0 cannot occur and a 0 column would be a comparison-shaped lie. **The stratum result is draw-sensitive where the aggregate is not, and that is reported rather than resolved:** at seed 42 the 50+ band reads "not distinguishable" and at seed 43 it reads "item-item wins, p = 0.002". Two draws are not a distribution, so the honest reading of that band is *item-item at least as good and plausibly better*, not a settled ordering — and the safe claim is the direction, which both draws agree on. **Read this against L34.** ALS earns its reputation on thin anchors; the demo declines to serve thin anchors by design. *This does not say ALS is the wrong engine* — it says the engine choice has no measured advantage behind it in the regime it runs in, and possibly a measured disadvantage, which is a different and weaker claim than the one six documents made | 2026-08-09 |
| L82 | **What makes ALS's neighbourhoods good is the support floor, not the factorization — and the floor costs five-sixths of its reach** | ALS **0.0308** with the L34 floor of 20, **0.0130** without it: Δ = **+0.0177** [+0.0151, +0.0204], 295 wins / 54 losses, **p = 2.5e-41** — the largest effect anywhere in this section, larger than any difference between two models. Without the floor ALS scores **below the popularity baseline** (0.0130 against 0.0155). The floor's price is reach: **AnchorCoverage@10 3.139% with it, 23.021% without**, against item-item's 17.262% | Identical model, identical factors, identical anchors; only `similar_min_support` changes (`als.py`). L34 found the floor on three anchors in M8 and its own headline was *"ALS needs a support floor"* — which has been read as *"ALS wins"* ever since. Both halves are now priced on 13,580 anchors: the floor is worth **more than the model choice**, and it buys that by refusing to answer with anything the 196k single-interaction items could have supplied | 2026-08-09 |
| L83 | **Every model is worse when the query shrinks from a history to one book, and the collaborative models lose the most** | Profile → anchor, share of the profile column retained, with paired McNemar: item-item 0.0644 → 0.0297 (**46.2%**, 152/622, p = 3.1e-68), item-item explicit-only 0.0486 → 0.0208 (42.9%, 70/447, p = 3.0e-68), ALS 0.0545 → 0.0308 (**56.5%**, 204/526, p = 1.3e-33), TF-IDF 0.0405 → 0.0258 (**63.8%**, 153/352, p = 4.3e-19), embeddings 0.0141 → 0.0126 (89.5%, 124/144, **p = 0.246, not distinguishable**), popularity 0.0155 → 0.0155 (0 discordant) | Same readers, same held-out books, only the query changes. The ordering of the retention column is the mechanism: a content model's query was **always** a single text vector, so shrinking the input costs it least; a collaborative model built on co-occurrence loses most of its evidence when a whole shelf becomes one book. **The number the demo has to own:** the rule this project recommends loses more than half its accuracy when the query shrinks — cascade **0.0650 → 0.0317**, against the published table's own leader at 0.0644 (L53). *When written this was also the best item-to-item row in the section; **L85 (M22) has since measured RRF at 0.0371 and fusion at 0.0355**, so the superlative is retired while the ratio it carried is unchanged — the best item-to-item row in this section is still barely half the best profile row (0.0371 against 0.0690). Scoped 2026-08-10.*. "No login and no reading history" is the right product decision for this demo and it has a measured price, and that price is not visible anywhere in the table the sidebar prints | 2026-08-09 |

**L80 · sensitivity, which is not a result.** Re-drawn at seed 43 the ordering of all eight rows
is **identical** and the largest cell movement is **+0.0016** (cascade, 0.0317 → 0.0333;
item-item **+0.0015**, ALS +0.0006, TF-IDF −0.0005, embeddings −0.0001, popularity and the ALS
ablation unchanged to four decimals), all well inside the ~0.004 this split resolves (L74). The
headline comparison gets *less* separable, not more: **ALS against item-item 178/175, p =
0.915**. The aggregate is therefore stable under the free parameter; **one stratum is not** —
see L81, where the same second draw moves the 50+ band from p = 0.167 to p = 0.002, and which
reports that rather than picking the friendlier draw. *Three figures in this paragraph were
corrected on 2026-08-10 by M22, which re-ran both draws: the largest movement is +0.0016 not
+0.0015, item-item did not hold still but moved +0.0015, and ALS moved +0.0006 not +0.0007.
Every measured cell of both draws reproduced to the digit against the stored artefacts, so it
was a reading error in the prose rather than a reproduction failure, and the claim it supports
— identical ordering, every movement inside the resolution — is unchanged.*

### The two rules that were missing from that table (milestone M22)

M19 measured three combination rules on the profile query and the accuracy winner there is
**score fusion** (L77), not the backfill cascade (L76). M20 then measured the item query — and
`measure_anchor_hitrate.py` built exactly one hybrid, the cascade. So the eight rows above held
the *conservative* rule and not the *winning* one, and the honest answer to **"your best model
isn't in your product table — why?"** was "we didn't measure it". This is that gap closed.
Nothing is re-fitted: all three rules are the same wrapper over the same two fitted base models,
and the run **asserts** all eight published rows above reproduce to the digit before it reports
anything (it does; `PUBLISHED_ANCHOR` in the script, non-zero exit if not).

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L85 | **The accuracy winner does win the item query too — and every point of its win sits in the two bands the demo refuses to serve** | AnchorHitRate@10 over the same **13,580 anchors**: **RRF 0.0371** [0.0341, 0.0404] (504 hits), **score fusion 0.0355** [0.0325, 0.0387] (482 hits), against cascade 0.0317, ALS 0.0308, item-item 0.0297. **AnchorCoverage@10 23.359% and 23.398%**, against the cascade's 21.357% and item-item's 17.262%; Novelty 16.82 / 16.80. Paired McNemar — **against item-item**: RRF 152/52, Δ = +0.0074 [+0.0053, +0.0094], p = 1.5e-12; fusion 119/41, Δ = +0.0057 [+0.0039, +0.0076], p = 5.2e-10. **Against the cascade** (the rule these would replace): RRF 123/50, Δ = +0.0054, p = 2.8e-08; fusion 90/39, Δ = +0.0038, p = 8.3e-06. **Against ALS** (the row six documents credited): RRF 250/164, Δ = +0.0063, p = 2.8e-05; fusion 234/170, Δ = +0.0047, p = 0.002. **RRF against fusion: 56/34, Δ = +0.0016 [+0.0003, +0.0030], p = 0.026** — the parameter-free rule is *ahead of* the tuned one here, where on the profile query the two could not be told apart at all (162/158, p = 0.867, L77). **But by the anchor's train support, RRF against item-item: 1–4 → 53/3 (p = 8.1e-13), 5–49 → 58/7 (p = 4.3e-11), 50+ → 41/42 (p = 1.000).** Against the cascade in that same 50+ band: **41/42, p = 1.000**. The app's anchor floor is 50 readers (L65) | `python scripts/measure_anchor_hitrate.py` — M22 replaced the hard-wired cascade with `--rules` (default cascade + rrf + fusion). Same anchors, same readers, same held-out books, same row order, `hit_vector` unchanged; the eight L80 rows reproduce to the digit in the same run, asserted. **α = 0.6 is inherited from L77's tuning on the *profile* query's inner validation split (seed 43) and was NOT re-tuned for the item query** — re-tuning needs a second inner draw plus a sweep, and L77's α curve is a broad plateau from 0.6 to 0.8 rather than a peak; the row is labelled rather than left to read as fitted. Candidate depth is 100 per base model, unchanged and binding here exactly as in L77. Coverage of the rules and the second draw are below the table | 2026-08-10 |

**L85 · why both new rules answer every anchor, and what a second draw does to them.**
`answered` is **100% for both new rules, not the 89.6% the milestone predicted**, and the
prediction was wrong for an interesting reason: the *content half* is silent on the 10.42% of
anchors with no catalogue row, but the collaborative half answers, so on those **1,415 anchors
both rules simply are item-item**. **Seed sensitivity, per L80's precedent:** re-drawn at seed
43 the ordering is identical (RRF 0.0360 > fusion 0.0351 > cascade 0.0333 > ALS 0.0314 >
item-item 0.0312), RRF moves −0.0011 and fusion −0.0004, and every comparison holds except one
— **RRF against fusion becomes 52/40, p = 0.251, not distinguishable**, so "RRF beats fusion"
is the one claim here that does not survive a second draw. The 50+ band moves the *other* way
on that draw: RRF against item-item **30/46, p = 0.085**, fusion against item-item **17/35,
p = 0.018 — distinguishably worse**.

**The retention column, in L83's form.** Profile → anchor, share retained: **RRF 0.0687 →
0.0371 (54.0%)**, **fusion 0.0690 → 0.0355 (51.4%)**, cascade 0.0650 → 0.0317 (48.8%). Both new
rules lose about half their hit rate when the query shrinks from a shelf to one book, which puts
them between the collaborative models (46.2%) and TF-IDF (63.8%) — the position L83's mechanism
predicts for a rule that is half collaborative and half content, and the first row in that table
that was not predictable from a single model class.

**Read the ledger cell as two sentences, because the two halves point opposite ways.** The
accuracy winner of the profile table wins the item query as well, by the widest margin in this
section — RRF adds **+25% relative** over item-item and **+17%** over the cascade, both far
outside the ±0.004 this split resolves. And in the band the demo actually serves, **41 wins
against 42 losses**: nothing. Every point of the win is bought in the 1–4 and 5–49 bands, and
L65 pins the app's anchor floor at 50 readers, so the demo would be paying two models at
serving time for a gain it has no anchors to collect. **This is L81's finding with a different
model in it** — ALS's advantage lived in the same place, and the app declines to serve there by
design. Twice now, a rule that wins the aggregate wins it where the product does not look.

**And the second half of "best neighbourhoods" already has a number for these two rules, from
M19.** L79 hand-counted the three-anchor gallery — which *is* this item-to-item surface — at
**cascade 0 bad slots of 30, fusion 11 of 30, RRF 13 of 30**: RRF answers *The Lovely Bones*
with *The Lovely Bones*, *Desde Mi Cielo* and *In meinem Himmel*. So the rule that now leads the
predictive column is the same rule with the worst-looking lists in the case. M20 said "best
neighbourhoods" was always a claim about both halves; for RRF the two halves now disagree as
sharply as they ever have in this project.

**What M20 does and does not settle.** It measures whether a neighbourhood is *predictive*
under an item query. It does **not** measure whether one *looks sensible* to a reader — that
is what the gallery and L47/L59's hand count do, and L79 is the standing proof that the two
can disagree: score fusion wins on a metric and answers *The Lovely Bones* with the same book
in three languages. "Best neighbourhoods" was always a claim about both halves. After M20 the
first half has a number and it does not support a superlative for any model; the second half
still ties item-item and ALS at 0 bad slots of 30.

**The consequence for the story, stated plainly.** The engine choice is now *defensible*
rather than *evidenced*: on the demo's own question, in the demo's own operating band, ALS is
not measurably better or worse than the model that wins the published table. Switching the
demo to item-item would cost nothing measurable and would remove an explanation from the
sidebar; leaving it on ALS costs nothing measurable either. **That is an open decision,
and this section exists so it is taken on numbers rather than on a sentence from 04.08.**

## Every number is conditional on one draw (milestone M21)

L74 answers "how certain is this" for one of the two sources and says so: its Wilson intervals
and paired tests hold the drawn holdout **fixed** and quantify sampling across *readers*. Seed
42 also decides **which** of a reader's favourites is withdrawn, and the paragraph under that
table names it as the larger uncovered source and records that it was not done. This section
does it, on **five draws — 42, 44, 45, 46, 47**. *Seed 43 is skipped deliberately: it already
names the inner validation split in L51 and L77, and two different draws sharing one number is
how a reader ends up believing a sweep tuned on its own test set.*

**The claim under test is the ordering, not the level.** L73 has item-item at 0.0000 / 0.0170 /
0.0571 / 0.1198 across the support bands — a factor of seven, and the draw decides which band a
reader's book lands in. The models are strong in *different* bands, so a deeper draw does not
move all six rows down together, it moves them **against each other**. Five levels tell you the
number wobbles; five paired deltas tell you whether the ordering does.

Command: `python scripts/measure_seed_sensitivity.py` (`--drift-only` runs the gate alone).

**The answer, in one sentence:** *the ordering holds on all five draws — every comparison keeps
its sign — but three near-ties change their significance verdict between draws, and one of them
is ALS against item-item, which L74 published as separable at p = 2.7e-06 on the strength of
seed 42 alone.*

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L84 | **The embedding cache cannot detect the drift it exists to detect: it checks 0.22% of the catalogue and missed a real change on four seeds out of four** | Changing the seed moves the canonical title of **76–80 works** per draw (**0.032–0.034%** of 235,824). Across the sweep that is 313 drift *events* on **167 distinct works** (**0.0708%**) — the overlap is the point: the works that flip are the ones whose two best editions have near-equal support, so the same ones flip on almost every draw (*About a Boy* on all four, *A Fortunate Life* and *A Man Called Intrepid* on two each). The cause is that `work_level_catalog` picks each work's text from its **most-interacted edition counted on train only**, and a single withheld book flips that argmax — e.g. *about a boy (movie tie-in) nick hornby* → *about a boy nick hornby*. `_fingerprint` hashes the model name, the array length and every `len//512`-th text: **513 of 235,824 positions, 0.2175%**. Expected catches for a 78-work drift: **0.17**. Observed: **0 of 4**. The fingerprint was **identical** (`db505f300fedb737`) on all five seeds | `python scripts/measure_seed_sensitivity.py --drift-only`, diffing the full text array with no model in the loop — never the fingerprint, which is the thing under test. **This is a stale cache *hit*, not a miss, and that is the dangerous direction:** a plain multi-seed run scores each seed with the *first* seed's vectors and reports it as a measurement. Blast radius and the decision not to fix it are below the table | 2026-08-09 |
| L86 | **The ordering survives all five draws. Three near-ties do not, and one of them is the pair the case leans on** | **Sign: all eight comparisons against item-item keep their sign on every draw — no row of the table ever changes place.** Levels (min–max, sd over 5 draws): RRF **0.0672–0.0687** (sd 0.0006), fusion α=0.5 0.0644–0.0660 (0.0006), cascade 0.0606–0.0650 (0.0016), **item-item 0.0599–0.0644** (0.0017), **ALS 0.0545–0.0585** (0.0016), explicit-only 0.0459–0.0501 (0.0016), TF-IDF 0.0385–0.0412 (0.0010), popularity 0.0133–0.0155 (0.0010), embeddings 0.0133–0.0142 (0.0003). So the published cell is the **top of the range** for item-item and the **bottom** for ALS. **Where distinguishability moves:** ALS against item-item is separable on **4 of 5** draws (Δ −0.0099 / −0.0049 / **−0.0023, p = 0.292** / −0.0050 / −0.0063) — L74's p = 2.7e-06 is seed 42's draw and the same pair is a coin-toss on seed 45; cascade against item-item on **4 of 5**, with Δ = **+0.0007 on all five draws** and only p crossing 0.05 (0.078 at seed 47); fusion α=0.5 against item-item on **1 of 5**. **What survives every draw:** RRF against item-item, **5 of 5**, Δ +0.0043 to +0.0082. And embeddings against popularity is **not distinguishable on 5 of 5** (p 0.184–0.718) with the sign flipping twice — L74's one published non-result, confirmed the hard way | `python scripts/measure_seed_sensitivity.py`. Seeds **42, 44, 45, 46, 47**; **43 skipped deliberately** because it names the inner validation split in L51/L77, and two draws sharing a number is how a reader concludes a sweep tuned on its own test set. **Asserted, not assumed:** all five draws score the *same* 13,580 readers — pinned by `tests/test_split.py::test_eligibility_is_seed_independent`, and the script exits non-zero if it fails, which is what makes these a paired sample rather than five unrelated experiments. **Seed 42 reproduces the published table, all three metrics and every per-user hit vector bit-for-bit** against M18's cache, so the sweep is not measuring a moved baseline. The two failure modes and three caveats are below the table | 2026-08-10 |

**L84 · blast radius, checked rather than assumed.** Every published number is seed 42, whose
cache is self-consistent, and the shipped app calls `work_level_catalog` with **no holdout**
(`scripts/build_app_assets.py:90`), so it is seed-independent and unaffected. The exposure is
multi-seed experiments — i.e. exactly this milestone, which is why the gate ran before anything
was fitted. **Not fixed here, deliberately:** hashing the full array is the right fix and it
invalidates every cached set, forcing an unplanned ~2-hour re-encode at a point where that time
was not available. Logged as the first item of the next session; M21's sweep instead **reuses
the cache knowingly and bounds the error**, reporting per seed how many held-out books carry a
stale vector.

**L86 · the two failure modes, read apart.** A *sign* flip would mean the table is in the wrong
order on some draw and would trigger M21's escalation rule — that did not happen. A *verdict*
change means a near-tie crossed p = 0.05, which is a caveat on the sentence that calls the pair
separable, not on the ordering. The script reported them as one thing on its first run and was
corrected before this line was written.

**L86 · three caveats, none of them discovered afterwards.** (1) The fusion row is **α = 0.5,
the constructor default, not L77's α = 0.6** — a third fusion variant, *not* comparable to L77;
margin 2 is therefore answered through **RRF, which is parameter-free** and needs nothing
inherited from seed 42. The script now defaults `--fusion-alpha` to 0.6 and labels it as
inherited. (2) Hyperparameters are **not** re-tuned per seed: λ=10 / 50 neighbours and ALS 128 /
α=1 were chosen on validation carved from seed 42's train, so under seed 47 an item now held out
took part in that sweep. Real, small — both sweeps landed on plateaus, and L51 is the null
result showing the choice does not depend on its split — and 5× the sweep cost to close. (3) The
embeddings rows reuse a **knowingly stale cache** (L84): 54–62 of 13,580 held-out books carry a
vector computed from different text, a worst-case bound of **±0.0046**, larger than the
embeddings-against-popularity gap it would have to adjudicate — a second and independent reason
that one comparison stays unanswerable here. Observed movement of the embeddings row across the
five draws is 0.0009, well inside that bound.

## The floor and the engine are one decision (milestone M23)

L81 and L85 found the same shape twice: **no engine this project has measured is separated
*ahead* of item-item above the app's anchor floor, and every separation that would justify a
different engine sits below it.** "Indistinguishable above the floor" would be the tidier
sentence and it is not what the two rows say: on their second draw both turn *against* the
challenger there — item-item ahead of ALS at p = 0.002 (L81) and ahead of fusion at p = 0.018
(L85). The direction is what both draws agree on, and the direction is the whole argument. That makes "which
engine" and "where is the floor" one question rather than two, and M23 is that question given
three measurements. Three configurations are named throughout: **A** = ALS, what ships;
**B** = item-item's shrunk cosine; **C** = RRF over item-item + TF-IDF, L85's winner.

*L86 is deliberately unused — reserved for M21's five-seed sweep, which was still running when
these were written. A gap is cheaper than the two id collisions this ledger has already had.*

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L87 | **L63's anti-calibration is a property of ALS, not of the data — and at the same floor item-item halves the thin-slot rate** | Median co-readers behind a shown slot against the median displayed number, anchor-support bands 20-30 → 600+ (60 anchors per band, seed 42, 3,190 slots per engine). **ALS: evidence 3.0 → 27.0 (×9) while the number moves 0.507 → 0.354, −30% — anti-calibrated, L63 reproduced exactly.** **item-item: 5.0 → 119.0 (×24) while the number moves 0.113 → 0.177, +56% — calibrated in the right direction.** RRF: 4.0 → 93.5 (×23), 0.016 → 0.018 (+10%). Thin slots (<5 co-readers) in the lowest band: ALS **76.2%**, RRF 62.8%, item-item **49.3%**. The floor pricing and the zero-co-reader counts are below the table | `python scripts/analyze_anchor_support.py --engines als item-item rrf --floors 20 50 --per-band 60`. Measured on the app's own assets with nothing re-fitted — the claim is about what a visitor sees, so it is measured on what a visitor queries (the M14.1 rule, unchanged). The engine seam is `recommender/engines.py`; the **candidate** floor stays at 20 in every row (L34), so one control moves one variable. **The displayed numbers are not comparable across engines and are not compared:** A is an ALS factor cosine, B a shrunk cosine, C a fused rank sum — which is why C's configuration carries an empty score label rather than a dressed-up one. **What this licenses and what it does not:** it says the *evidence behind a shown slot* is better under B at either floor; it says nothing about which list is more predictive (L88) or which looks sensible (L89) | 2026-08-10 |
| L88 | **The band a lower floor would open cannot separate the engines, and the evidence that looked like it could sits in the band that stays shut** | The identical cached per-anchor hit vectors of L80/L85, re-cut at **5 / 20 / 50** and printed beside L73's boundaries. The **20-49** band is **1,860 anchors — 13.7% of all anchors and 27.9% of the ones a floor of 20 would make askable**. AnchorHitRate@10 there (seed 42): RRF 0.0419, item-item 0.0328, cascade 0.0328, ALS 0.0317, TF-IDF 0.0312. Paired McNemar in that band, **both draws**: **RRF against item-item 20/3 (p = 4.9e-04) at seed 42 but 9/4 (p = 0.267) at seed 43 — not replicated**; ALS against item-item **20/22 (p = 0.878) and 20/15 (p = 0.500) — nothing in either draw**. What *does* replicate is one band lower: RRF against item-item in **5-19** is 38/4 (p = 5.7e-08) and 33/7 (p = 4.2e-05). So L85's 58/7 across 5-49 is carried by the half a floor of 20 still closes | `python scripts/recut_anchor_bands.py`. Nothing re-scored, nothing re-fitted: the hit is `eval.hit_vector` exactly as in L80, L81 and L85, and the grouping functions moved into `recommender/eval.py` so the re-cut reads the same code as the rows it must stay comparable with. Published boundaries printed alongside, never instead. **The 20-49 band is underpowered, and that is the finding rather than a caveat**: at ~3% hit rates it holds 23 and 13 discordant readers in the two draws, so it cannot settle an engine question — it can only fail to. **L81's warning arriving on schedule**: a stratum whose two draws disagree is reported unsettled, not resolved by taking the friendlier draw | 2026-08-10 |
| L89 | **On the band the floor would open, the engine that wins the metric shows books with no shared readers six times as often as the one that ships — and item-item is the only configuration that never does** | Face-validity audit: **8 anchors drawn uniformly from the 20-49 band × 3 configurations × 10 slots = 240 slots, every one read by hand** (`docs/floor_band_audit.md`). Bad slots under **L79's criterion, inherited unchanged**: **A 1 of 80, B 0 of 80, C 2 of 80**. A second column, defined in M23 because L79's criterion cannot see this band's dominant failure — a *text-match artefact* shares a title word or **a name token from the anchor author's name (first or last)**, carries at most one co-reader, and is not a comparable read: **A 0, B 0, C 17 of 80 (21.3%)**. **Slots with literally zero shared readers, over the same 240: A 2, B 0, C 12** — so C does it six times as often as the shipped engine, and B never does it at all. The mechanism behind all seventeen is below the table | Counted **by hand**, 2026-08-10, all 240 slots, the way L42's 30 clusters and L79's 30 gallery slots were counted; every counted slot is named in the doc, so the count can be checked rather than believed. **The counterweight is recorded in the same place:** on *Beauty: A Retelling of the Story of Beauty and the Beast*, C returns five Robin McKinley novels including *Rose Daughter*, her other Beauty-and-the-Beast retelling — the best single slot any configuration produces in the table, at **0 co-readers**, which no collaborative engine could ever find. C's worst failure and C's best moment are the same mechanism pointed at different anchors | 2026-08-10 |

**L87 · pricing the anchor floor at 20 against 50**, i.e. 7,541 askable works and 39.7%
interaction coverage against 2,508 and 26.6% (L65): thin-slot share at floor 20 is ALS
**49.8%**, RRF 39.0%, **item-item 25.3%**, against ALS's **18.2%** at the shipped floor of 50;
median co-readers 5.0 / 6.0 / **7.0** against 10.0.

**L87 · who shows slots with no shared readers at all.** Only RRF does it at a material rate:
**10.7%** of slots in the 20-30 band and 2.1% even at 600+. item-item is **0.0% in all six
bands**; ALS is 0.0% in three of them and 0.2-0.3% in the other three — small, but not never.
The first version of this line said "0.0% for both others", which was reading a rounded column
as a universal (corrected 2026-08-10, see L89).

**L89 · the seventeen text-match artefacts are one mechanism.** C answers *Thank You for
Smoking* by Christopher **Buckley** with two Christopher **Pike** novels at zero co-readers,
*Songlines* by **Bruce** Chatwin with **Bruce** Coville and **Bruce** Sterling, and *Main
Street* by Sinclair **Lewis** with C. S. **Lewis**. Both strict failures are the work key:
*Songlines* returning *The Songlines* (a leading article it does not strip — in A **and** C),
and C returning one Ellen DeGeneres book under two punctuations.

**The verdict these three support, as a recommendation and not a change.** The shipped
configuration A is the only one of the three that is *anti-calibrated* — it prints its highest
numbers where its evidence is thinnest (L87) — and its 50-reader floor exists to hide exactly
that. B removes the pathology instead of hiding it: at the **same** floor of 20 it halves the
thin-slot rate (25.3% against 49.8%), lifts the median evidence per slot from 5 to 7 co-readers,
never returns a zero-evidence slot, and produced **no bad slot and no artefact** in 80 hand-read
ones. C wins the predictive column (L85) and loses on everything a reader would actually see.

**So the floor is doing more work than the model choice, which is what M23 set out to find.**
Moving the anchor floor from 50 to 20 under **B** would take the askable catalogue from 2,508
works to **7,541** and interaction coverage from 26.6% to **39.7%**, at a thin-slot rate in the
same region as the one the demo ships with today. Nothing was switched: the demo still runs A at
floor 50, every published number is untouched, and this is the first item of the next session
rather than an instruction taken at 3am (M23 decisions 11 and the standing M20/M22 rule).

## The switch, activated on two configurations at one floor (milestone M23.10)

M23.10 ships **B beside A as a picker, with A the default**. C is dropped and does not come
back, for L89's reason. **Both configurations sit at anchor floor 50** — decided on
10.08., reversing the milestone's first draft — so the click moves **one** variable: the same
anchor, the same 2,508 askable works, a calibrated similarity instead of an anti-calibrated one.
The floor's own cost is demonstrated separately and with no picker in it, by typing *The Kite
Runner* and being declined.

**Nothing below is a model result and nothing published moves.** This is the serving layer:
the two lines are a *gate* on what a visitor sees and a *cost* of showing it.

| ID | Claim | Number | How measured | Measured |
|---|---|---|---|---|
| L90 | **B survives the read A was never given: 0 bad slots in 220 against A's 7 — and the same twenty anchors show the work key splitting more books than anyone had counted, including the demo's own first anchor** | **440 slots read by hand**: 20 anchors × 2 configurations × 10 = 400, plus 2 anchors the search box cannot reach (`docs/anchor_set_audit.md`). Bad slots under **L79's criterion, inherited unchanged: A 7 of 220 (4 unambiguous), B 0 of 220.** The script's mechanical column over the same 440: **median co-readers A 15, B 46; thin slots (<5) A 42 (19.1%), B 6 (2.7%); zero-co-reader slots 0 for both** — reproducing **L87's 18.2% against 1.7%** on a different anchor set and a different sampling rule | `python scripts/audit_anchor_set.py` — writes the table and counts the co-reader column, and **does not grade the slots**: the face-validity count is filled in by hand under section 5, every counted slot named so it can be re-checked rather than believed, the way L42's 30 clusters, L79's 30 gallery slots and L89's 240 were counted. Counted **by hand**, 2026-08-10. **The twenty are reached by typing the title into the app's own `find` path**, not by work id, because the gate is about what a visitor gets; where a query does not land on the book the set names, both anchors are read. Reader counts come from this command, not from a session (the M16 rule) | 2026-08-10 |
| L91 | **The switch costs nothing that can be measured on a stopwatch: B's whole answer table is 0.22 MB, it builds in 22 seconds, and configuration A comes back byte-identical on all 110 rehearsed slots** | **A unchanged:** the eleven rehearsed anchors × 10 = **110 slots** against a from-first-principles recomputation of the pre-seam engine — work ids in order, **scores to twelve decimals**, co-reader counts, same-author tags and reason sentences — **identical through both entry points**. **B's table:** the **2,508** askable works (L65's filter: above the anchor floor *and* nameable, not the 2,532 above the floor alone) × top-10 = **25,080 rows, 0.22 MB, 8.9 bytes a row**, built in **21.8 s** (9 ms an anchor), 0 anchors short of ten slots; re-read from disk and re-compared against a live recomputation of every anchor, **0 lists differ, worst score deviation 2.9e-08** — float32 rounding, four decimal digits below anything the app prints. **Cold start against L69's 10.6 s: A 8.8 s, B 8.5 s**; warm query A 21 ms, B 22 ms; assets **890 MB**, unchanged, because the table is additive | `python scripts/verify_configuration_a.py` (the 110 slots; exits non-zero on any difference), `python scripts/build_answer_table.py` (builds, re-reads and self-verifies against the live source in one run), `python scripts/measure_app_latency.py --configuration A` and `--configuration B`. **No published number moves. This is serving** | 2026-08-10 |

**L90 · the seven slots, named.** A's four unambiguous failures are one defect: *The Hobbit*
returning **the anchor under another title**, and *To Kill a Mockingbird*, *Love in the Time of
Cholera* and *War and Peace* each showing **one book twice** under two work keys a leading
article apart. Three more are companion volumes (*Fantastic Beasts*, *Quidditch Through the
Ages*, *The Tolkien Reader*), counted and flagged because L34's own docstring treats that class
as a result worth keeping. **B cannot make the duplicate mistake structurally**: two halves of a
split work share almost no readers, so their co-occurrence is near zero, while A's factors put
them close together — the engine that reasons in a latent space inherits the catalogue's
duplicates, and the one that counts shared readers is blind to them.

**L90 · the anchor set**, with every reader count printed by the command rather than read off a
screen: The Lovely Bones 1,295 · The Da Vinci Code 905 · Harry Potter and the Sorcerer's Stone
832 · Bridget Jones's Diary 772 · Life of Pi 658 · Girl with a Pearl Earring 647 · Interview
with the Vampire 521 · To Kill a Mockingbird 495 · Tuesdays with Morrie 492 · Angela's Ashes
326 · The Hobbit 281 · Love in the Time of Cholera 261 · Dune 257 · One Hundred Years of
Solitude 252 · The Curious Incident 204 · Crime and Punishment 140 · Fight Club 102 · War and
Peace 100 · Guns, Germs, and Steel 67 · The Master and Margarita 65; plus **The Kite Runner
39**, below the floor and pinned as the floor demonstration.

**L90 · the work-key splits**, counted over the whole nameable catalogue rather than instanced:
**10,737 groups / 23,091 works** in four classes — subtitle 5,543, leading article 2,210,
internal punctuation 1,043, parenthetical 88, plus 1,853 residual. **194 of those groups are
silenced at the floor**: every half below 50 while the sum clears it, which is **+7.7% on the
askable catalogue** available without touching the floor or the engine. Named instances: *The
Brothers Karamazov* **36 + 26 = 62** (leading article; together askable, separately not — L89's
*Songlines* defect a second time), *The Curious Incident* **204 + 82**, *Angela's Ashes* **326 +
283 + 224** (three ways, not two), *The Hobbit* **281 + 123 + 112 + 1 + 1**, and **the demo's
own first anchor, The Lovely Bones, 1,295 + 103**.

**L90 · two things the gate found that are not about the engine.** (1) **2 of 20 titles do not
resolve to the book they name** — "Guns, Germs, and Steel" answers with *Secrets* by Danielle
Steel and "War and Peace" with *Peace Like a River*, because `LOOKUP_TIE_MARGIN` (L62) demotes
the correct top text match by readership when four Danielle Steel novels sit within 0.06 of it.
That is L38's under-determined query on a famous title, engine- and floor-independent, and
**not fixed here**: the tie rule is published, a rehearsed demo runs on it, and the right book
is offered at picker rank 4 and rank 3. (2) **The milestone's representativeness note was wrong
and is corrected here.** *The Purpose Driven Life* does **not** have 1 reader:
`the purpose-driven life…\|warren` has **79** and is askable. The 1-reader row is
`the purpose driven life…\|warren`, the same book under a second work key one **hyphen** away —
a statement about the key, not about the sample.

**L91 · what the answer table is, and what it is not.** **B's live similarity is 9 ms an anchor
against A's 44 ms**, so the table is not a latency fix and is not offered as one: it is the
Part 3 Gold-table serving pattern built rather than drawn, pinning answers to a build instead
of recomputing them inside a request. **Additive by construction** (M23 decision 7): nothing in
`artifacts/app/` was rewritten, B is one new 0.22 MB file beside the shipped arrays, keyed on
the work key already stamped in `meta.json`, and a stamp mismatch **refuses the load** rather
than answering with the right numbers attached to the wrong books. The stamp covers the item
count, the work key, both floors and a **sha256 of the support vector** — the only one of the
five that can see a rebuild landing on the same shape — and `tests/test_answers.py` proves each
of the five can stop a load.

**What the two lines license, and what they do not.** L90 says B's lists are *safe to put on
screen* and better-evidenced than A's on this set; it does **not** say B is more predictive —
L85 and L88 own that column and neither separates the two above the floor. L91 says the switch
is free; it says nothing about which setting is right. **A stays the default**, one constant in
`app/main.py` removes the picker, and the reason A is default is that it is the rehearsed
engine, not that it won anything.

**Where A is genuinely better, recorded because a switch has to be honest in both directions.**
A is more *specific* where B is more *evidenced*: nine Anne Rice novels under *Interview with
the Vampire* against B's six, a Vermeer-and-Dutch-painting tail under *Girl with a Pearl
Earring*, a science-fiction tail under *Dune*, and a Russian tail under *Crime and Punishment*
where B answers with the English canon. B's two weakest slots in 220 are *Jurassic Park* under
*Interview with the Vampire* (92 shared readers) and *The Da Vinci Code* under *One Hundred
Years of Solitude* (58): **generic, not wrong**, and the count is printed beside the row. A's
weakest is a book **three of 658 readers** share, at **rank 1** of *Life of Pi*.

## Open items this ledger will need

- ~~Edition clustering in the data-prep layer (L31, L39)~~ — **done, M11 (L40–L47).**
  ~~What remains is a decision: does the whole comparison table move to work
  level?~~ — **decided and done, M12 (L49–L59).** All six models are re-run on the work
  basis, and L58 explains why the lift was not uniform across them.
- ~~**Re-tune item-item on a work-level validation split**~~ — **done, L51: the sweep
  re-selected λ=10 and 50 neighbours, so L44's "+18% is a lower bound for this reason" is
  retired.** The +18% stands on its own.
- **Beyond title equality** (L47, L59): translations and alternate titles still defeat the
  clustering, and only the text-based models suffer — 7 of 30 gallery slots for TF-IDF and
  6 of 30 for embeddings, on the work basis. Either LLM metadata enrichment or an external
  work identifier; both also address the lookup failure in L38.
- ~~The hybrid itself measured, rather than argued from the L50 ceiling~~ — **done, M19
  (L76–L79, 2026-08-09).** Three rules, one recommended (the backfill cascade), and the
  prediction written before the run marked falsified-as-stated and held-in-its-reasoning.
  **What it opened in turn**, both narrower than the item it closes: (a) the fusion rules'
  duplicate problem is the L47 edition/translation ceiling arriving in a new place, so
  fixing that clustering is now the *first* thing to re-measure fusion against rather than a
  general improvement; (b) a cascade with a support floor between 5 and 20 is where the
  trade turns negative, and the crossing point is unmeasured — L76 has floors 0, 5 and 20
  and nothing between. **A third thing it opened is now closed: L85 (M22, 2026-08-10)** puts
  the two fusion rules on the item query, so the product table no longer holds only the
  conservative rule. What *that* opens is narrower again: RRF's α-free win is real on the
  aggregate and absent in the served band, so the open question is no longer "which rule" but
  **"is the app's 50-reader floor the right place to stand"** — every rule this project has
  measured is separated below it and indistinguishable above it.
- Item-item normalized by profile length, to see whether it removes the long-profile
  degradation in L28. **Still open, and L73 sharpens it rather than closing it**: at work
  level the degradation is 0.0716 at 10–24 items against 0.0370 at 75+, and ALS over the same
  users loses far less (0.0568 → 0.0445), which points at the summation rule rather than at
  long profiles being intrinsically hard.
- ~~**The strata in L27 and L28 have not been recomputed on the work basis** *as ledger
  lines*~~ — **done, L73 (2026-08-09).** Both strata, all six models, from the same per-user
  hit vectors as L74. The state this item described — the notebook's §2.1 printing numbers
  the ledger did not carry — is what L73 closes.
- ~~**No uncertainty is quantified anywhere in this ledger.**~~ — **done, L74
  (2026-08-09).** 95% Wilson intervals on all six cells and paired McNemar over the per-user
  hit vectors, every model against item-item and against the baseline. The one comparison
  this item said the ledger did not survive — popularity against embeddings — is the one the
  test declines to order (p = 0.342), and the wording it had already been given on 09.08 was
  right. **What is still open is narrower**: the ISBN-level table has had no such treatment,
  and Coverage and Novelty have no intervals at all. Coverage is not a per-user proportion,
  so the same machinery does not apply to it — a bootstrap over users would, and it is
  unmeasured.
- ~~**The holdout draw itself is unmeasured, and it is now the largest unquantified
  uncertainty in the project.**~~ — **measured, M21 (L86).** Five draws (42, 44, 45, 46, 47)
  over the same 13,580 readers. **The ordering holds on every one of them**; what moves is the
  significance verdict on three near-ties, ALS against item-item among them. Every number here
  is still conditional on one draw — that has not changed — but the *consequence* of that is
  now bounded rather than open: sd 0.0003 to 0.0017 per row, and no rank ever moves. **What
  remains open from this item** is the tuning channel L86 names and does not close:
  hyperparameters were selected on seed 42's validation split and inherited by the other four,
  which is 5x the sweep cost to fix.
- **The subtitle class of duplicate works** (L64's note): 9,523 further works across 30,486
  clusters would merge if everything after a colon were dropped. Needs an M11.3-style
  sampled audit before anyone touches it, because the rule cannot tell *The Hobbit: or
  There and Back Again* (merge) from *Bridget Jones: The Edge of Reason* (do not).
- ~~**The app and the published table now use different work keys** (L64, a deliberate decision).
  The difference is priced — item-item +0.8%, denominator −0.5% — and recorded in
  `artifacts/app/meta.json`, but it is a divergence and a slide has to be able to say so.~~
  **Closed 2026-08-09: it does not go in the write-up.** The divergence stays exactly
  where it is — priced in L64, stamped into `artifacts/app/meta.json`, and explained here —
  and it is answered if it is asked, not volunteered. Recorded as a decision rather than
  deleted, so nobody re-opens it: at +0.8% on one row, seven users of 13,580 and well inside
  the ±0.0021 standard error, it is not a finding, and a slide spent on it would buy
  precision nobody asked for at the cost of the minute that carries the argument.
- **L67 has no counterpart for the other models.** Neighbourhood stability was measured for
  the ALS surface the app serves, because that is where it was noticed. Whether item-item's
  neighbourhoods are steadier under the same perturbation is unmeasured and would be a
  cheap, genuinely useful comparison.

---

**Honesty note that travels with every number above.** These are offline,
single-dataset measurements on a 2004 crawl. They describe the data, and later they will
describe model behaviour on a held-out slice of it. They are a proxy for whether
recommendations are *good* — a live A/B test is the only real proof, and that stays true
however favourable the offline table looks.
