# ReviewLens

Sentiment classification and complaint analysis of Jollibee Foodpanda reviews.

## Folder layout

```
ReviewLens/
  data/jollibee_reviews_raw.csv   <- put your raw export here (or pass its path)
  text_prep.py                    shared tokenizer / stopwords / negations
  preprocess.py        NEW        load, inspect, clean, label, split
  eda.py               NEW        exploratory analysis + figures
  train_grid.py                   24-config grid search + AFINN baseline
  evaluate.py          NEW        evaluation report without the dashboard
  extract_complaints.py FIXED     complaint terms + regex themes + branch lift
  validate_themes.py   NEW        measures how accurate the regex themes are
  run_all.py           NEW        runs everything in order
  check_tagalog.py     FIXED      scratch check (not part of the pipeline)
  app.py               FIXED      Streamlit dashboard - NOT for the midterm
  output/                         everything the scripts write
```

## Run order

```bash
pip install -r requirements.txt
python run_all.py data/jollibee_reviews_raw.csv
```

or one at a time:

| Step | Command | Writes |
|---|---|---|
| 1 | `python preprocess.py [raw.csv]` | `jollibee_train/test/complaints/branch_summary.csv`, `preprocessing_report.json` |
| 2 | `python eda.py` | `output/eda/*.png` |
| 3 | `python train_grid.py` | `cv_results.csv`, `predictions_test.csv`, `best_model.joblib`, `best_config.json` |
| 4 | `python evaluate.py` | `output/eval/*` (confusion matrix, per-class scores, CV chart, errors, top words) |
| 5 | `python extract_complaints.py` | `complaint_terms_*.csv`, `complaint_themes_branch.csv`, `branch_priorities.csv` |
| 6 | `python validate_themes.py examples` / `sample` / `score` | `theme_validation_*.csv` |

`preprocess.py` finds the raw columns by name (e.g. `text`/`review`/`comment`,
`rating`/`overall`/`stars`, `vendor_name`/`branch`). If it says it can't find
a column, add the raw name to `COLUMN_CANDIDATES` at the top of the file.

## What changed in existing files

**extract_complaints.py:** narrowed theme patterns that matched unrelated complaints:
- dropped bare `wala`/`walay` (Missing item)
- request words now need a contrast word like "but"/"pero" to count as Wrong order
- chicken parts only count after "ordered"/"asked for"
- dropped bare `hours`/`minutes` (Late delivery)
- "hard to …" no longer counts as Food quality
- dropped `cup` and `basa` (Packaging)

The full list is in the file's docstring. **Theme numbers will change after you re-run.**

**app.py:** the 48.8% baseline, the "Feb 2024 to Apr 2025" range and the "~10% Neutral"
figure were typed in by hand. They're now calculated from the data.

**train_grid.py:** the closing line used to always say Neutral was the weakest
class. It now reports whichever class actually is.

**check_tagalog.py:** the Windows-only absolute path is now relative, it uses
the project tokenizer, and it matches whole words, so "mali" no longer matches "maliit".
