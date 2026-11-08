"""Quick check: which Tagalog complaint words actually appear in the complaints?

Used while writing the theme regexes in extract_complaints.py. Not part of the
main pipeline.  Run:  python check_tagalog.py

Fixes vs the first version:
- path is relative to this file (worked only on one laptop before)
- uses the project tokenizer, so Tagalog stopwords go and negations stay,
  exactly as the model sees the text
- matches whole words: "mali" no longer matches "maliit", "wala" no longer
  matches "walang"
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from text_prep import normalize, tokenize

OUT = Path(__file__).parent / "output"
TAGALOG = {"kulang", "mali", "walang", "malamig", "sobrang", "tagal", "hilaw",
           "matigas", "bawas", "wala", "hindi", "masama", "maalat", "tuyo",
           "sirang", "panis", "matagal", "dumating", "tamad", "bastos"}

df = pd.read_csv(OUT / "jollibee_complaints.csv", encoding="utf-8-sig")
vec = CountVectorizer(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                      lowercase=False, ngram_range=(1, 2), min_df=3)
X = vec.fit_transform(df["text_clean"].fillna("").astype(str))
terms = pd.DataFrame({"term": vec.get_feature_names_out(),
                      "n_reviews": (X > 0).sum(axis=0).A1})

mask = terms["term"].map(lambda t: any(w in TAGALOG for w in t.split()))
print(terms[mask].sort_values("n_reviews", ascending=False).head(30).to_string(index=False))
