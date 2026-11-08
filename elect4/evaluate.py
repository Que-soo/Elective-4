"""ReviewLens - Part 4: evaluation report for the sentiment model (no dashboard).

Run after train_grid.py:  python evaluate.py
Reads output/best_config.json, cv_results.csv, predictions_test.csv and
best_model.joblib, prints the evaluation and saves figures/tables to output/eval/.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

import text_prep  # noqa: F401  (needed so best_model.joblib can find tokenize())

OUT = Path(__file__).parent / "output"
EVAL = OUT / "eval"
LABELS = ["Negative", "Neutral", "Positive"]
TOP_K = 15


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(EVAL / name, dpi=150)
    plt.close(fig)
    print(f"  saved output/eval/{name}")


def top_features(model, k=TOP_K):
    """Words that push the model toward each class. For linear models this is the
    coefficient; for Naive Bayes it is how much more likely a word is in that
    class than in the other two (log-probability difference)."""
    vec, clf = model.named_steps["vec"], model.named_steps["clf"]
    names = vec.get_feature_names_out()
    if hasattr(clf, "coef_"):
        weights = clf.coef_
    else:
        lp = clf.feature_log_prob_
        weights = np.array([lp[i] - np.delete(lp, i, axis=0).mean(axis=0) for i in range(len(lp))])
    rows = []
    for i, cls in enumerate(clf.classes_):
        for rank, j in enumerate(np.argsort(weights[i])[::-1][:k], 1):
            rows.append({"class": cls, "rank": rank, "term": names[j], "weight": float(weights[i, j])})
    return pd.DataFrame(rows)


def main():
    EVAL.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((OUT / "best_config.json").read_text(encoding="utf-8"))
    cv = pd.read_csv(OUT / "cv_results.csv", encoding="utf-8-sig")
    pred = pd.read_csv(OUT / "predictions_test.csv", encoding="utf-8-sig")
    train = pd.read_csv(OUT / "jollibee_train.csv", encoding="utf-8-sig")
    y, p = pred["sentiment"], pred["predicted"]

    # ---------------------------------------------------------------- 1. experiments
    banner("1. EXPERIMENT COMPARISON (5-fold cross-validation on the training set)")
    show = cv[["rank", "config", "macro_f1_mean", "macro_f1_std", "accuracy_mean"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ml = cv[cv["vectorizer"] != "AFINN lexicon"]
    print("\nAverage macro-F1 by design choice (mean over the other settings):")
    for col in ("vectorizer", "ngram_range", "model"):
        print(f"  {col}:")
        for k, v in ml.groupby(col)["macro_f1_mean"].mean().sort_values(ascending=False).items():
            print(f"    {k:<22} {v:.3f}")
    afinn = cv[cv["vectorizer"] == "AFINN lexicon"]
    best = cv[cv["is_best"]].iloc[0]
    if len(afinn):
        print(f"\nBest model vs AFINN lexicon baseline: {best['macro_f1_mean']:.3f} vs "
              f"{afinn['macro_f1_mean'].iloc[0]:.3f} macro-F1")
    runner_up = ml.iloc[1] if len(ml) > 1 else None
    if runner_up is not None:
        gap = best["macro_f1_mean"] - runner_up["macro_f1_mean"]
        print(f"Gap to runner-up ({runner_up['config']}): {gap:.3f} "
              f"(fold std of the best = {best['macro_f1_std']:.3f}"
              f"{' - within noise, models are effectively tied' if gap < best['macro_f1_std'] else ''})")

    # ---------------------------------------------------------------- 2. test set
    banner(f"2. HELD-OUT TEST SET - {cfg['config']}")
    acc = (y == p).mean()
    majority = cfg["majority_baseline"]
    print(f"Test accuracy  {acc:.1%}   (majority baseline, always '{majority['label']}': "
          f"{majority['accuracy']:.1%})")
    print(f"Test macro-F1  {f1_score(y, p, labels=LABELS, average='macro'):.3f}   "
          f"(CV estimate {cfg['cv_macro_f1_mean']:.3f} ± {cfg['cv_macro_f1_std']:.3f})")
    print()
    print(classification_report(y, p, labels=LABELS, digits=3, zero_division=0))
    cm = confusion_matrix(y, p, labels=LABELS)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=[f"actual {l}" for l in LABELS],
                       columns=[f"pred {l}" for l in LABELS]).to_string())
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(EVAL / "confusion_matrix.csv")

    # ---------------------------------------------------------------- 3. robustness checks
    banner("3. ROBUSTNESS CHECKS")
    seen = pred["text_clean"].str.lower().isin(set(train["text_clean"].str.lower()))
    rows = [("All test reviews", np.ones(len(pred), bool)),
            ("Text never seen in training", ~seen.to_numpy()),
            ("Text also in training set", seen.to_numpy()),
            ("Short (< 3 words)", pred["is_short"].astype(bool).to_numpy()),
            ("Longer (3+ words)", ~pred["is_short"].astype(bool).to_numpy())]
    sub = []
    for name, m in rows:
        if m.sum() == 0:
            continue
        sub.append({"subset": name, "n": int(m.sum()), "accuracy": (y[m] == p[m]).mean(),
                    "macro_f1": f1_score(y[m], p[m], labels=LABELS, average="macro", zero_division=0)})
    sub = pd.DataFrame(sub)
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("If 'never seen' scores close to 'all', duplicated short texts are not inflating the result.")
    sub.to_csv(EVAL / "subset_scores.csv", index=False)

    # ---------------------------------------------------------------- 4. errors
    banner("4. ERROR ANALYSIS")
    wrong = pred[~pred["correct"]].copy()
    pairs = wrong.groupby(["sentiment", "predicted"]).size().sort_values(ascending=False)
    print(f"{len(wrong)} of {len(pred)} test reviews misclassified. Most common mistakes:")
    for (a, b), n in pairs.items():
        print(f"  actual {a:<8} -> predicted {b:<8} {n:>4}")
    sample = (wrong.sample(frac=1, random_state=42)
              .groupby(["sentiment", "predicted"]).head(5)
              .sort_values(["sentiment", "predicted"]))
    sample[["sentiment", "predicted", "rating", "text_clean"]].to_csv(
        EVAL / "misclassified_examples.csv", index=False, encoding="utf-8-sig")
    print("\nExamples (full list in output/eval/misclassified_examples.csv):")
    for _, r in sample.head(8).iterrows():
        print(f"  [{r['rating']}* {r['sentiment']} -> {r['predicted']}] {str(r['text_clean'])[:90]}")

    # ---------------------------------------------------------------- 5. what the model learned
    banner("5. WHAT THE MODEL LEARNED (top words per class)")
    model = joblib.load(OUT / "best_model.joblib")
    feats = top_features(model)
    feats.to_csv(EVAL / "top_features.csv", index=False, encoding="utf-8-sig")
    for cls, g in feats.groupby("class"):
        print(f"  {cls:<9} " + ", ".join(g["term"].head(12)))

    # ---------------------------------------------------------------- figures
    print("\nFigures:")
    fig, ax = plt.subplots(figsize=(5, 4.2))
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "black", fontsize=10)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set(xlabel="Predicted", ylabel="Actual (from star rating)", title="Confusion matrix (test set)")
    save(fig, "1_confusion_matrix.png")

    prf = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(3)
    for k, (name, vals, c) in enumerate(zip(["Precision", "Recall", "F1"], prf[:3],
                                            ["#2a78d6", "#eb6834", "#1baf7a"])):
        bars = ax.bar(x + (k - 1) * 0.26, vals, 0.26, label=name, color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, LABELS)
    ax.set(ylim=(0, 1.1), title="Per-class scores (test set)")
    ax.legend(ncol=3, loc="lower center", fontsize=8)
    save(fig, "2_per_class_scores.png")

    fig, ax = plt.subplots(figsize=(8, 0.3 * len(cv) + 1))
    c = cv.iloc[::-1]
    colors = ["#eb6834" if b else ("#c3c2b7" if v == "AFINN lexicon" else "#2a78d6")
              for b, v in zip(c["is_best"], c["vectorizer"])]
    ax.barh(c["config"], c["macro_f1_mean"], xerr=c["macro_f1_std"], color=colors,
            error_kw=dict(lw=0.8, capsize=2))
    ax.set(xlabel="Macro-F1 (5-fold CV mean ± std)", xlim=(0, 1),
           title="All configurations (orange = selected, grey = AFINN baseline)")
    ax.tick_params(axis="y", labelsize=7)
    save(fig, "3_cv_comparison.png")

    summary = {"test_accuracy": float(acc), "test_macro_f1": float(f1_score(y, p, labels=LABELS, average="macro")),
               "majority_baseline": majority, "subsets": sub.to_dict("records"),
               "error_pairs": {f"{a}->{b}": int(n) for (a, b), n in pairs.items()}}
    (EVAL / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("  saved output/eval/evaluation_summary.json, confusion_matrix.csv, subset_scores.csv, "
          "misclassified_examples.csv, top_features.csv")


if __name__ == "__main__":
    main()
