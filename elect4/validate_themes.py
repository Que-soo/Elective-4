"""ReviewLens - check how accurate the regex complaint themes are.

The themes in extract_complaints.py are rules we wrote, so they need to be
checked against human judgement. Three steps:

  python validate_themes.py examples
      Runs hand-written test sentences through the regexes (quick sanity check
      you can show live).

  python validate_themes.py sample [N]
      Draws N random complaints (default 150) into
      output/theme_validation_labels.csv with EMPTY theme columns. Open it in
      Excel and put 1 in every theme a complaint is really about (leave blank
      otherwise). The regex answers are saved separately in
      output/theme_validation_key.csv so the labeller isn't influenced by them.

  python validate_themes.py score
      Compares your labels with the regex tags: precision (when the regex says
      "Late delivery", how often is it right?), recall (of the truly late
      complaints, how many did it catch?) and F1, per theme.
"""
import sys
from pathlib import Path

import pandas as pd

from extract_complaints import THEME_LABELS, THEME_RE, theme_text

OUT = Path(__file__).parent / "output"
LABELS_FILE = OUT / "theme_validation_labels.csv"
KEY_FILE = OUT / "theme_validation_key.csv"
SCORES_FILE = OUT / "theme_validation_scores.csv"
SEED = 42

# (sentence, set of themes it SHOULD get)
EXAMPLES = [
    ("walang gravy po", {"MISSING_ITEM"}),
    ("kulang yung rice ko", {"MISSING_ITEM"}),
    ("wala pa rin po yung order after 1 hour", {"LATE_DELIVERY"}),
    ("I requested thigh but got wings", {"WRONG_ORDER"}),
    ("the leg part was dry", {"FOOD_QUALITY"}),
    ("mali ang order ko, spaghetti instead of palabok", {"WRONG_ORDER"}),
    ("sobrang tagal dumating", {"LATE_DELIVERY"}),
    ("it took hours", {"LATE_DELIVERY"}),
    ("malamig na yung chicken", {"COLD_FOOD"}),
    ("the rice is hard", {"FOOD_QUALITY"}),
    ("rider was hard to contact", set()),
    ("a cup of rice was missing", {"MISSING_ITEM"}),
    ("di nagbasa ng notes", set()),
    ("spilled ang sauce, crushed yung box", {"PACKAGING"}),
    ("may buhok sa spaghetti", {"HYGIENE"}),
    ("rude cashier", {"STAFF_SERVICE"}),
    ("puro buto ang chicken", {"SMALL_PORTION"}),
    ("lateeee na nga malamig pa", {"LATE_DELIVERY", "COLD_FOOD"}),
]


def tag(text):
    t = theme_text(text)
    return {k for k, rx in THEME_RE.items() if rx.search(t)}


def examples():
    ok = 0
    for text, want in EXAMPLES:
        got = tag(text)
        good = got == want
        ok += good
        mark = "PASS" if good else "FAIL"
        print(f"[{mark}] {text!r}\n        expected {sorted(want) or '-'}  got {sorted(got) or '-'}")
    print(f"\n{ok}/{len(EXAMPLES)} examples tagged exactly as expected.")


def sample(n):
    df = pd.read_csv(OUT / "jollibee_complaints.csv", encoding="utf-8-sig")
    s = df.sample(min(n, len(df)), random_state=SEED)[["uuid", "Branch", "text_clean"]]
    labels = s.copy()
    labels.insert(0, "checked", "")
    for k in THEME_RE:
        labels[k] = ""
    labels.to_csv(LABELS_FILE, index=False, encoding="utf-8-sig")
    key = s[["uuid"]].copy()
    for k, rx in THEME_RE.items():
        key[k] = s["text_clean"].map(lambda t, rx=rx: int(bool(rx.search(theme_text(t)))))
    key.to_csv(KEY_FILE, index=False)
    print(f"Wrote {len(s)} complaints to {LABELS_FILE}.\n"
          "For each row: put 1 under every theme the complaint is really about (blank = no),\n"
          "and put 1 in 'checked' once you've read it. Save as CSV, then run:\n"
          "  python validate_themes.py score")


def score():
    if not LABELS_FILE.exists() or not KEY_FILE.exists():
        sys.exit("Run 'python validate_themes.py sample' and fill in the labels first.")
    human = pd.read_csv(LABELS_FILE, encoding="utf-8-sig")
    key = pd.read_csv(KEY_FILE)
    themes = list(THEME_RE)
    filled = human["checked"].notna()
    if not filled.any():
        sys.exit(f"No rows in {LABELS_FILE.name} are marked as checked yet.")
    human = human[filled]
    print(f"Scoring {len(human)} checked complaints ({(~filled).sum()} not checked yet, skipped).\n")
    m = human[["uuid"] + themes].merge(key, on="uuid", suffixes=("_true", "_pred"))
    rows = []
    for k in themes:
        t = pd.to_numeric(m[f"{k}_true"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        p = m[f"{k}_pred"].astype(int)
        tp, fp, fn = int(((t == 1) & (p == 1)).sum()), int(((t == 0) & (p == 1)).sum()), int(((t == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        rows.append({"theme": THEME_LABELS[k], "human_yes": int(t.sum()), "regex_yes": int(p.sum()),
                     "correct": tp, "false_alarms": fp, "missed": fn,
                     "precision": prec, "recall": rec, "f1": f1})
    res = pd.DataFrame(rows)
    micro_tp, micro_fp, micro_fn = res["correct"].sum(), res["false_alarms"].sum(), res["missed"].sum()
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    mp, mr = micro_tp / max(micro_tp + micro_fp, 1), micro_tp / max(micro_tp + micro_fn, 1)
    print(f"\nAll themes pooled: precision {mp:.2f}, recall {mr:.2f}")
    print("Themes with fewer than ~10 human 'yes' labels have unreliable scores - say so.")
    res.to_csv(SCORES_FILE, index=False)
    print(f"Saved {SCORES_FILE}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "examples"
    if mode == "examples":
        examples()
    elif mode == "sample":
        sample(int(sys.argv[2]) if len(sys.argv) > 2 else 150)
    elif mode == "score":
        score()
    else:
        sys.exit(__doc__)
