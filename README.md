# Book Recommender — research project on the Book-Crossing dataset

A personal research project: build a book recommender end to end on the
[Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
— explore the data, try the natural modelling approaches, evaluate them honestly, and
end in an interface where you paste a book and get recommendations back, with an
explanation of *why* those books.

Optimized for a clear, honest, demoable story and for learning something — not for
shipping a product. Offline metrics are treated as a proxy; a live A/B test would be
the real proof.

The work runs in three stages:

1. **Data understanding** — what this dataset actually is, and which modelling decisions
   it forces. See [`docs/dataset_findings.md`](docs/dataset_findings.md) and
   [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).
2. **Modelling and evaluation** — a ladder of candidate models on one pinned split, each
   measured on accuracy, catalogue coverage and novelty. See
   [`notebooks/02_models.ipynb`](notebooks/02_models.ipynb), the write-up
   [`docs/model_selection.md`](docs/model_selection.md) and the ledger
   [`docs/RESULTS.md`](docs/RESULTS.md).
3. **Interface and productionization** — paste a book, get recommendations with an
   explanation; plus how such a system would scale and stay fresh on a data platform.

## Getting the data

The raw CSVs are **not** in this repo (they are large, and licensed by their source).
Download them from Kaggle — dataset `arashnic/book-recommendation-dataset` — and place
the three files in `data/`:

```
data/Books.csv     # ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-*  (~271k)
data/Ratings.csv   # User-ID, ISBN, Book-Rating (0-10)                                            (~1.15M)
data/Users.csv     # User-ID, Location, Age                                                       (~279k)
```

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle datasets download -d arashnic/book-recommendation-dataset -p data --unzip
```

All files are comma-separated. Note that `Ratings.csv` has **no timestamp column** — this
shapes how evaluation splits are done (see [`docs/dataset_findings.md`](docs/dataset_findings.md)).

## Environment setup

Python 3.11, in a virtual environment:

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

To run the notebooks in this environment, register its Jupyter kernel once — otherwise
`jupyter` may silently execute them against a different interpreter:

```bash
python -m ipykernel install --user --name recommender --display-name "Python 3.11 (recommender)"
```

Then: `pytest` for the test suite, `ruff check .` for lint, and the model runner to
reproduce the **primary** comparison table. The item is a *work*, not an ISBN — editions of
the same book are merged before the split (see `docs/RESULTS.md` L49):

```bash
python scripts/run_model.py --all --gallery --work-level
```

Drop `--work-level` to reproduce the ISBN-keyed table instead. That one is kept in the
ledger as the journey record; the two are **not** cell-comparable, because they have
different items and different coverage denominators.

The edition-clustering work (same book, many ISBNs) has its own entry points:

```bash
python scripts/analyze_editions.py --write-sample docs/edition_clusters_sample.md
```

```bash
python scripts/analyze_dedup.py
```

And the measurement that explains why the two tables differ by so much more for the text
models than for the collaborative ones (ledger L58):

```bash
python scripts/decompose_work_level_lift.py
```

## Repository layout

| Path | What lives there |
|---|---|
| `data/` | The raw Book-Crossing CSVs (not committed — see above). |
| `notebooks/` | The journey: `01_eda.ipynb` (data understanding), `02_models.ipynb` (model comparison). |
| `docs/` | [`RESULTS.md`](docs/RESULTS.md) — the measurement ledger — plus [`dataset_findings.md`](docs/dataset_findings.md), the model-selection write-up [`model_selection.md`](docs/model_selection.md), and figures under `docs/img/`. |
| `src/recommender/` | One module per responsibility: data prep, split, models, evaluation, gallery. |
| `scripts/` | Entry points: `run_model.py` (evaluate models), `tune_*.py` (hyperparameters on a validation split), `analyze_editions.py` and `analyze_dedup.py` (the edition-clustering measurements), `decompose_work_level_lift.py` (why the work-keyed table differs from the ISBN-keyed one). |
| `tests/` | Offline, deterministic tests — no network, no model downloads. |

[`docs/RESULTS.md`](docs/RESULTS.md) is the measurement ledger: every headline number used
anywhere in this project traces to a line there, including the negative results.

## Status

Data understanding and the model comparison are done. Six models — popularity baseline,
item-item CF (plus an explicit-only ablation), ALS, content TF-IDF and multilingual
sentence embeddings — are fitted on one pinned leave-one-out split and measured on
HitRate@10, Coverage@10 and Novelty@10; the table and every negative result behind it are
in [`docs/RESULTS.md`](docs/RESULTS.md), and the reasoning behind the choice is written up
in [`docs/model_selection.md`](docs/model_selection.md). The interface and the
productionization write-up follow.

The published table is keyed by **work** rather than by ISBN — merging editions before
training is the largest single accuracy gain in the project, and it is a data-preparation
change rather than a model one. The ISBN-keyed table is kept beside it as the journey
record, and the reason the two differ far more for the text models than for the
collaborative ones is measured rather than asserted (ledger L58).

## References

The sources this project leans on, in full in
[`docs/dataset_findings.md`](docs/dataset_findings.md):

- Ziegler et al., WWW 2005 — [Improving Recommendation Lists Through Topic Diversification](https://doi.org/10.1145/1060745.1060754),
  the paper the Book-Crossing dataset comes from.
- Naghiaei, Rahmani, Deldjoo, 2022 — [The Unfairness of Popularity Bias in Book Recommendation](https://arxiv.org/abs/2202.13446),
  the popularity-bias benchmark on this dataset.
- Hu, Koren, Volinsky, ICDM 2008 — [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22),
  the weighted-MF formulation behind the ALS model, used via [`implicit`](https://github.com/benfred/implicit).
- Steck, WWW 2019 — [Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375) (EASE).
- [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  — the sentence-embedding model used for the content layer.

## License

The code and documentation in this repository are released under the MIT License — see
[`LICENSE`](LICENSE). The Book-Crossing data is **not** covered by it: it is not part of
this repo and stays under the terms of its own source (see
[Getting the data](#getting-the-data)).
