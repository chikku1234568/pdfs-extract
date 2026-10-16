"""Compare outputs/summary.csv with your hand-filled manual_review.csv.

    python -m kyctrial.evaluate --pred outputs/summary.csv --truth outputs/manual_review.csv
Only rows where a *_true cell is filled are scored."""
import argparse
import pandas as pd
from rapidfuzz import fuzz


def _norm(x):
    return " ".join(str(x).lower().split()) if pd.notna(x) else ""


def evaluate(pred_csv, truth_csv, name_fuzzy=90):
    pred = pd.read_csv(pred_csv, dtype=str)
    truth = pd.read_csv(truth_csv, dtype=str)
    df = truth.merge(pred, on="file", how="left")
    out = []
    for field in ["has_statements", "months_covered", "app_id", "name", "co_app_name", "co_app_id"]:
        tcol = field + "_true"
        sub = df[df[tcol].notna() & (df[tcol].str.strip() != "")]
        if sub.empty:
            continue
        hits = []
        for _, r in sub.iterrows():
            t, p = _norm(r[tcol]), _norm(r.get(field))
            if field == "has_statements":
                ok = (t in ("1", "true", "yes", "y")) == (p in ("1", "true", "yes", "y"))
            elif field == "months_covered":
                ok = p != "" and abs(float(t) - float(p)) < 0.5
            elif field in ("name", "co_app_name"):
                ok = fuzz.ratio(t, p) >= name_fuzzy
            else:
                ok = t.replace(" ", "") == p.replace(" ", "")
            hits.append(ok)
        out.append({"field": field, "n": len(hits), "correct": sum(hits),
                    "accuracy": round(sum(hits) / len(hits), 3)})
    res = pd.DataFrame(out)
    print(res.to_string(index=False) if not res.empty else "No filled *_true cells found.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/summary.csv")
    ap.add_argument("--truth", default="outputs/manual_review.csv")
    a = ap.parse_args()
    evaluate(a.pred, a.truth)


if __name__ == "__main__":
    main()
