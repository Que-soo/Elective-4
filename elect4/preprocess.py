"""ReviewLens - Part 0: data loading, inspection, cleaning and train/test split.

Turns the raw review export into the four files every other script reads:

  output/jollibee_train.csv           80% of clean reviews (model training + CV)
  output/jollibee_test.csv            20% held out, touched once by train_grid.py
  output/jollibee_complaints.csv      every Negative review (complaint analysis)
  output/jollibee_branch_summary.csv  one row per branch (counts, % negative)
  output/preprocessing_report.json    every number printed below, for the slides

Run:  python preprocess.py                       (uses RAW_PATH below)
      python preprocess.py path/to/raw_reviews.csv

The raw file only needs a text column and a star-rating column; branch, city,
date and review id are picked up when present. Column names are matched
case-insensitively against COLUMN_CANDIDATES - add your own names there if the
script says it can't find a column.
"""
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
OUT = HERE / "output"
RAW_PATH = HERE / "data" / "jollibee_reviews_raw.csv"

SEED = 42
TEST_SIZE = 0.20
SHORT_WORDS = 3          # fewer than this many words -> is_short
MIN_BRANCH_REVIEWS = 30  # branches below this are flagged as too small to compare

# Our name -> names the column might have in the raw export (first match wins).
COLUMN_CANDIDATES = {
    "uuid": ["uuid", "review_id", "reviewid", "id"],
    "Branch": ["branch", "branch_name", "vendor_name", "vendor", "restaurant",
               "restaurant_name", "store", "store_name"],
    "City": ["city", "location", "area"],
    "review_date": ["review_date", "date", "created_at", "createdat", "timestamp",
                    "review_time", "updated_at", "updatedat"],
    "rating": ["rating", "overall", "overall_rating", "rating_overall", "score",
               "stars", "star_rating"],
    "text_raw": ["text_raw", "text", "review", "review_text", "comment", "content",
                 "body", "message"],
}
REQUIRED = ["text_raw", "rating"]

# A complaint that mentions the rider is about delivery, not the branch's kitchen.
RIDER_RE = re.compile(
    r"\b(?:rider|riders|driver|courier|delivery\s+(?:guy|man|boy|person|personnel|partner)"
    r"|kuya\s+rider)\b", re.IGNORECASE)
_ZERO_WIDTH_RE = re.compile("[​-‍⁠﻿]")
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W_]+")


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- 1. loading
def load_raw(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Raw file not found: {path}\nPut it there or pass the path: "
                 "python preprocess.py path/to/file.csv")
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    elif path.suffix.lower() == ".json":
        raw = pd.read_json(path)
    else:
        raw = pd.read_csv(path, encoding="utf-8-sig")

    lower = {c.lower().strip(): c for c in raw.columns}
    mapping = {}
    for ours, names in COLUMN_CANDIDATES.items():
        hit = next((lower[n] for n in names if n in lower), None)
        if hit is not None:
            mapping[hit] = ours
    missing = [c for c in REQUIRED if c not in mapping.values()]
    if missing:
        sys.exit(f"Could not find column(s) {missing} in {list(raw.columns)}.\n"
                 "Add the raw column name to COLUMN_CANDIDATES in preprocess.py.")
    df = raw[list(mapping)].rename(columns=mapping)
    return raw, df, mapping


# ---------------------------------------------------------------- 2. cleaning helpers
def clean_text(text):
    """Fix encoding leftovers without changing what the customer wrote:
    HTML entities (&amp;), full-width/compatibility characters, zero-width
    characters and runs of whitespace. Case, emoji and spelling are kept -
    lowercasing and stopwords happen later inside text_prep.py."""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def has_content(text):
    """At least one letter, digit or emoji - "." or "!!!" carries nothing."""
    return any(ch.isalnum() or unicodedata.category(ch) == "So" for ch in text)


def parse_rating(series):
    """'5', '5.0', '5 stars', ' 4 ' -> number; anything else -> NaN."""
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
                         errors="coerce")


def to_sentiment(rating):
    return np.select([rating <= 2, rating == 3], ["Negative", "Neutral"], "Positive")


def unify_spelling(series):
    """Group names that differ only in case/spacing ('SM  north', 'SM North') and
    give every member the most common spelling in that group."""
    s = series.astype("string").str.strip().str.replace(_SPACE_RE, " ", regex=True)
    key = s.str.lower()
    canonical = s.groupby(key).agg(lambda v: v.value_counts().idxmax())
    return key.map(canonical), int((s != key.map(canonical)).sum())


# ---------------------------------------------------------------- main
def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else RAW_PATH
    OUT.mkdir(exist_ok=True)
    report = {"raw_file": str(raw_path)}

    # ---- 1. load + inspect
    raw, df, mapping = load_raw(raw_path)
    banner("1. DATASET LOADED")
    print(f"File: {raw_path}")
    print(f"Raw shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(f"Columns used: " + ", ".join(f"{k} -> {v}" for k, v in mapping.items()))
    print(f"Columns not used: {[c for c in raw.columns if c not in mapping]}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(max_colwidth=60))
    report.update(raw_rows=int(len(raw)), raw_columns=list(map(str, raw.columns)),
                  column_mapping=mapping)

    for col in ("uuid", "Branch", "City", "review_date"):
        if col not in df:
            df[col] = pd.NA

    banner("2. DATA QUALITY CHECK (before cleaning)")
    df["text_clean"] = df["text_raw"].map(clean_text)
    df["rating_num"] = parse_rating(df["rating"])
    empty_text = ~df["text_clean"].map(has_content)
    quality = {
        "missing_text": int(df["text_raw"].isna().sum()),
        "empty_or_symbol_only_text": int((empty_text & df["text_raw"].notna()).sum()),
        "missing_rating": int(df["rating"].isna().sum()),
        "unparseable_rating": int((df["rating_num"].isna() & df["rating"].notna()).sum()),
        "rating_out_of_range": int((~df["rating_num"].between(1, 5) & df["rating_num"].notna()).sum()),
        "missing_branch": int(df["Branch"].isna().sum()),
        "missing_city": int(df["City"].isna().sum()),
        "missing_date": int(df["review_date"].isna().sum()),
        "duplicate_review_id": int(df["uuid"].dropna().duplicated().sum()),
        "duplicate_full_rows": int((df.duplicated(["Branch", "review_date", "rating", "text_clean"])
                                    & df["review_date"].notna()).sum()),
        "same_text_different_review": int(df["text_clean"].str.lower().duplicated().sum()),
    }
    print("Missing values per column:")
    print(df[["uuid", "Branch", "City", "review_date", "rating", "text_raw"]]
          .isna().sum().to_string())
    print("\nIssues found:")
    for k, v in quality.items():
        print(f"  {k:<28} {v:>7,}")
    report["quality_before"] = quality

    # ---- 3. cleaning, one logged step at a time
    banner("3. CLEANING STEPS")
    steps = []

    def drop(mask, reason):
        nonlocal df
        n = int(mask.sum())
        df = df[~mask].copy()
        steps.append({"step": reason, "rows_removed": n, "rows_left": len(df)})
        print(f"  - {reason:<60} -{n:>6,}  -> {len(df):,} rows")

    drop(~df["text_clean"].map(has_content), "Drop reviews with no text (nothing to classify)")
    drop(df["rating_num"].isna(), "Drop reviews with no usable rating (no label)")
    drop(~df["rating_num"].between(1, 5), "Drop ratings outside 1-5 (invalid entries)")
    df["rating"] = df["rating_num"].round().astype(int)

    if df["uuid"].notna().any():
        drop(df["uuid"].notna() & df["uuid"].duplicated(), "Drop repeated review IDs (scraped twice)")
    # Needs a date to tell "scraped twice" from "two customers both wrote 'good'".
    drop(df.duplicated(["Branch", "review_date", "rating", "text_clean"]) & df["review_date"].notna(),
         "Drop exact duplicate rows (same branch, date, rating, text)")
    drop(df["Branch"].isna() | (df["Branch"].astype(str).str.strip() == ""),
         "Drop reviews with no branch (can't be assigned to a store)")

    df["Branch"], n_branch_fixed = unify_spelling(df["Branch"])
    df["City"] = df["City"].fillna("Unknown")
    df["City"], n_city_fixed = unify_spelling(df["City"])
    print(f"  - Unified branch spellings (case/spacing variants)              {n_branch_fixed:>7,} rows changed")
    print(f"  - Unified city spellings                                        {n_city_fixed:>7,} rows changed")

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
    n_bad_date = int(df["review_date"].isna().sum())
    print(f"  - Dates parsed to UTC; {n_bad_date:,} missing/unparseable kept "
          "(date is not a model input)")

    if df["uuid"].isna().any():  # make a stable id for rows that have none
        fill = pd.util.hash_pandas_object(df[["Branch", "text_clean", "rating"]], index=False)
        df["uuid"] = df["uuid"].astype("string").fillna(fill.astype(str).radd("gen_"))
    steps += [{"step": "Unify branch spellings", "rows_changed": n_branch_fixed},
              {"step": "Unify city spellings", "rows_changed": n_city_fixed},
              {"step": "Unparseable dates kept as missing", "rows": n_bad_date}]

    # ---- 4. derived columns
    banner("4. TARGET VARIABLE AND DERIVED COLUMNS")
    df["sentiment"] = to_sentiment(df["rating"])
    df["word_count"] = df["text_clean"].map(lambda t: len(_WORD_RE.findall(t)))
    df["is_short"] = df["word_count"] < SHORT_WORDS
    df["rider_blamed"] = (df["text_clean"].str.contains(RIDER_RE, na=False)
                          & (df["sentiment"] != "Positive"))
    branch_n = df["Branch"].map(df["Branch"].value_counts())
    df["branch_usable"] = branch_n >= MIN_BRANCH_REVIEWS

    print("Target = sentiment, derived from the star rating:")
    print("  1-2 stars -> Negative   3 stars -> Neutral   4-5 stars -> Positive")
    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())
    dist = df["sentiment"].value_counts()
    print("\nSentiment (class balance):")
    for lab in ["Negative", "Neutral", "Positive"]:
        print(f"  {lab:<9} {dist.get(lab, 0):>7,}  ({dist.get(lab, 0) / len(df):.1%})")
    wc = df["word_count"]
    q1, q3 = wc.quantile([0.25, 0.75])
    long_cut = q3 + 1.5 * (q3 - q1)
    print(f"\nWord count: median {wc.median():.0f}, mean {wc.mean():.1f}, max {wc.max()}")
    print(f"  Short reviews (< {SHORT_WORDS} words): {df['is_short'].sum():,} "
          f"({df['is_short'].mean():.1%}) - kept, flagged as is_short")
    print(f"  Long outliers (> {long_cut:.0f} words, IQR rule): {(wc > long_cut).sum():,} "
          "- kept; a long review is still a genuine review")
    print(f"Rider mentioned in a non-positive review: {df['rider_blamed'].sum():,}")
    print(f"Branches: {df['Branch'].nunique()} "
          f"({(df.groupby('Branch').size() >= MIN_BRANCH_REVIEWS).sum()} with "
          f">= {MIN_BRANCH_REVIEWS} reviews)   Cities: {df['City'].nunique()}")

    # ---- 5. split
    banner("5. TRAIN / TEST SPLIT")
    cols = ["uuid", "Branch", "City", "review_date", "rating", "sentiment", "text_raw",
            "text_clean", "word_count", "is_short", "rider_blamed", "branch_usable"]
    df = df[cols].reset_index(drop=True)
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["sentiment"],
                                   random_state=SEED)
    overlap = test["text_clean"].str.lower().isin(set(train["text_clean"].str.lower()))
    print(f"Stratified {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} split on sentiment, seed {SEED}")
    print(f"  train {len(train):,}   test {len(test):,}")
    print(pd.DataFrame({"train": train["sentiment"].value_counts(normalize=True),
                        "test": test["sentiment"].value_counts(normalize=True)})
          .map(lambda v: f"{v:.1%}").to_string())
    print(f"Test reviews whose exact text also appears in train: {overlap.sum():,} "
          f"({overlap.mean():.1%}) - mostly short generic reviews like 'good'; "
          "evaluate.py reports accuracy with and without them.")

    # ---- 6. save
    complaints = df[df["sentiment"] == "Negative"]
    g = df.groupby("Branch")
    summary = pd.DataFrame({
        "City": g["City"].agg(lambda c: c.value_counts().idxmax()),
        "n_reviews": g.size(),
        "n_negative": g["sentiment"].apply(lambda s: (s == "Negative").sum()),
        "n_neutral": g["sentiment"].apply(lambda s: (s == "Neutral").sum()),
        "n_positive": g["sentiment"].apply(lambda s: (s == "Positive").sum()),
        "avg_rating": g["rating"].mean().round(2),
    }).reset_index()
    summary["pct_negative"] = summary["n_negative"] / summary["n_reviews"]
    summary["usable"] = summary["n_reviews"] >= MIN_BRANCH_REVIEWS
    summary = summary.sort_values("n_reviews", ascending=False)

    write = lambda d, name: d.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    write(train, "jollibee_train.csv")
    write(test, "jollibee_test.csv")
    write(complaints, "jollibee_complaints.csv")
    write(summary, "jollibee_branch_summary.csv")

    report.update(
        cleaning_steps=steps, clean_rows=len(df),
        rows_removed=int(len(raw) - len(df)),
        rating_distribution={int(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()},
        sentiment_distribution={k: int(v) for k, v in dist.items()},
        word_count={"median": float(wc.median()), "mean": float(wc.mean()), "max": int(wc.max()),
                    "iqr_long_cutoff": float(long_cut), "n_long_outliers": int((wc > long_cut).sum())},
        n_short=int(df["is_short"].sum()), n_rider_blamed=int(df["rider_blamed"].sum()),
        n_branches=int(df["Branch"].nunique()), n_cities=int(df["City"].nunique()),
        n_usable_branches=int(summary["usable"].sum()),
        date_range=[str(df["review_date"].min()), str(df["review_date"].max())],
        split={"seed": SEED, "test_size": TEST_SIZE, "n_train": len(train), "n_test": len(test),
               "test_text_seen_in_train": int(overlap.sum())},
        n_complaints=len(complaints),
    )
    (OUT / "preprocessing_report.json").write_text(json.dumps(report, indent=2, default=str),
                                                   encoding="utf-8")
    banner("SAVED")
    print(f"{len(raw):,} raw rows -> {len(df):,} clean reviews "
          f"({len(raw) - len(df):,} removed)")
    print(f"{OUT}: jollibee_train.csv, jollibee_test.csv, jollibee_complaints.csv "
          f"({len(complaints):,} rows), jollibee_branch_summary.csv, preprocessing_report.json")


if __name__ == "__main__":
    main()
