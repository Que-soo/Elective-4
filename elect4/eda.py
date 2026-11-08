"""ReviewLens - exploratory data analysis on the cleaned reviews.

Run after preprocess.py:  python eda.py
Prints summary tables and saves figures to output/eda/ (show these in Part 2).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from text_prep import BISAYA_STOPWORDS, NEGATIONS, TAGALOG_STOPWORDS, normalize, tokenize

OUT = Path(__file__).parent / "output"
FIG = OUT / "eda"
SENTIMENTS = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#d03b3b", "Neutral": "#fab219", "Positive": "#0ca30c"}

# Words that only appear in one language's list - used to guess the review language.
TL_ONLY = (TAGALOG_STOPWORDS | {"hindi", "walang", "wala", "huwag", "wag"}) \
    - set(ENGLISH_STOP_WORDS) - BISAYA_STOPWORDS - {"o", "e", "un", "may"}
CEB_ONLY = (BISAYA_STOPWORDS | {"dili", "walay"}) - TAGALOG_STOPWORDS - set(ENGLISH_STOP_WORDS)
_WORDS = re.compile(r"[a-zñ]+")


def language_guess(text):
    words = set(_WORDS.findall(str(text).lower()))
    tl, ceb = bool(words & TL_ONLY), bool(words & CEB_ONLY)
    if ceb:
        return "Bisaya / mixed"
    if tl:
        return "Tagalog / Taglish"
    return "English only" if words else "Emoji / symbols only"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"  saved output/eda/{name}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(OUT / "jollibee_train.csv", encoding="utf-8-sig"),
                    pd.read_csv(OUT / "jollibee_test.csv", encoding="utf-8-sig")], ignore_index=True)
    df["review_date"] = (pd.to_datetime(df["review_date"], utc=True)
                         .dt.tz_convert("Asia/Manila").dt.tz_localize(None))

    print(f"Clean dataset: {len(df):,} reviews x {df.shape[1]} columns")
    print(df.dtypes.to_string())
    has_dates = df["review_date"].notna().any()
    if has_dates:
        print(f"\nDate range: {df['review_date'].min():%b %d, %Y} - {df['review_date'].max():%b %d, %Y}")
    else:
        print("\nNo review dates in this dataset - monthly chart skipped.")
    print(f"Branches: {df['Branch'].nunique()}   Cities: {df['City'].nunique()}")
    print("\nMissing values after cleaning:")
    na = df.isna().sum()[lambda s: s > 0]
    print(na.to_string() if len(na) else "  none")

    print("\nRating -> sentiment:")
    print(pd.crosstab(df["rating"], df["sentiment"]).to_string())

    df["language"] = df["text_clean"].map(language_guess)
    lang = df["language"].value_counts()
    print("\nLanguage mix (rough guess from function words):")
    for k, v in lang.items():
        print(f"  {k:<22} {v:>6,}  ({v / len(df):.1%})")

    print("\nWord count by sentiment:")
    print(df.groupby("sentiment")["word_count"].describe()[["mean", "50%", "max"]]
          .reindex(SENTIMENTS).round(1).to_string())
    print("\nShort reviews (< 3 words) by sentiment:")
    print(df.groupby("sentiment")["is_short"].mean().reindex(SENTIMENTS)
          .map("{:.1%}".format).to_string())

    for lab in SENTIMENTS:
        words = pd.Series([w for t in df.loc[df["sentiment"] == lab, "text_clean"]
                           for w in tokenize(normalize(t))])
        print(f"\nTop 10 words in {lab} reviews (stopwords removed, negations kept):")
        print("  " + ", ".join(f"{w} ({n})" for w, n in words.value_counts().head(10).items()))

    print("\nFigures:")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    counts = df["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values,
           color=[COLORS["Negative"]] * 2 + [COLORS["Neutral"]] + [COLORS["Positive"]] * 2)
    for x, v in zip(counts.index.astype(str), counts.values):
        ax.text(x, v, f"{v:,}\n({v / len(df):.0%})", ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="Star rating", ylabel="Reviews", title="Ratings are U-shaped (few 3-star reviews)")
    ax.margins(y=0.2)
    save(fig, "1_rating_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    s = df["sentiment"].value_counts().reindex(SENTIMENTS)
    ax.bar(s.index, s.values, color=[COLORS[c] for c in s.index])
    for x, v in zip(s.index, s.values):
        ax.text(x, v, f"{v:,} ({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Reviews", title="Target variable: class balance")
    ax.margins(y=0.15)
    save(fig, "2_class_balance.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cap = int(df["word_count"].quantile(0.99))
    for lab in SENTIMENTS:
        ax.hist(df.loc[df["sentiment"] == lab, "word_count"].clip(upper=cap),
                bins=[b - 0.5 for b in range(0, cap + 2)], alpha=0.55, label=lab, color=COLORS[lab])
    ax.axvline(2.5, color="#555", ls="--", lw=1)
    ax.text(2.7, ax.get_ylim()[1] * 0.9, "< 3 words = short", fontsize=8)
    ax.set(xlabel=f"Words per review (capped at {cap})", ylabel="Reviews",
           title="Review length by sentiment")
    ax.legend()
    save(fig, "3_review_length.png")

    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * df["Branch"].nunique() + 1)))
    bc = df.groupby(["Branch", "sentiment"]).size().unstack(fill_value=0).reindex(columns=SENTIMENTS, fill_value=0)
    bc = bc.loc[bc.sum(axis=1).sort_values().index]
    left = 0
    for lab in SENTIMENTS:
        ax.barh(bc.index, bc[lab], left=left, color=COLORS[lab], label=lab)
        left = left + bc[lab]
    ax.axvline(30, color="#555", ls="--", lw=1)
    ax.set(xlabel="Reviews (dashed line = 30, minimum for branch comparison)",
           title="Reviews per branch")
    ax.legend(loc="lower right")
    save(fig, "4_reviews_per_branch.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if not has_dates:
        plt.close(fig)
        fig = None
    m = None if fig is None else (df.dropna(subset=["review_date"])
         .assign(month=lambda d: d["review_date"].dt.to_period("M").dt.to_timestamp())
         .groupby(["month", "sentiment"]).size().unstack(fill_value=0).reindex(columns=SENTIMENTS, fill_value=0))
    if fig is not None:
        for lab in SENTIMENTS:
            ax.plot(m.index, m[lab], marker="o", ms=3, label=lab, color=COLORS[lab])
        ax.set(ylabel="Reviews", title="Monthly review volume")
        ax.legend()
        fig.autofmt_xdate()
        save(fig, "5_monthly_volume.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(lang.index[::-1], lang.values[::-1], color="#2a78d6")
    for y, v in zip(lang.index[::-1], lang.values[::-1]):
        ax.text(v, y, f" {v / len(df):.0%}", va="center", fontsize=9)
    ax.set(xlabel="Reviews", title="Language mix (why an English-only lexicon struggles)")
    ax.margins(x=0.15)
    save(fig, "6_language_mix.png")


if __name__ == "__main__":
    main()
