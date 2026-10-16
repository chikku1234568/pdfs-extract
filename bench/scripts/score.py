"""Score results/raw/*.json against data/gt.json -> results/results_per_page.csv, results/summary.csv"""
import csv, glob, json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import jiwer
from kyctrial.config import DEFAULTS
from kyctrial.ocr import group_lines, line_text
from kyctrial.statements import parse_page_table, reconcile

GT = json.load(open("data/gt.json"))
AMT = re.compile(r"^\d{1,3}(,\d{2,3})*\.\d{2}$")
DATE = re.compile(r"^\d{2}[/-](\d{2}|[A-Z][a-z]{2})[/-]\d{2,4}$")
KW = ("withdrawal", "deposit", "debit", "credit", "balance", "dr/cr")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def multiset_recall(gt_items, ocr_tokens):
    pool = Counter(ocr_tokens); hit = 0
    for g in gt_items:
        if pool[g] > 0: pool[g] -= 1; hit += 1
    return hit / len(gt_items) if gt_items else None


def score_page(gt, words):
    lines = group_lines(words) if words else []
    text = "\n".join(line_text(l) for l in lines)
    toks = [t.strip(" |:;") for t in text.split()]
    gtt = gt["text"]
    r = {"cer": jiwer.cer(norm(gtt), norm(text) or "x"), "wer": jiwer.wer(norm(gtt), norm(text) or "x")}
    gtoks = gtt.split()
    r["amount_recall"] = multiset_recall([t for t in gtoks if AMT.match(t)], toks)
    r["date_recall"] = multiset_recall([t for t in gtoks if DATE.match(t)], toks)
    if gt["type"] == "statement":
        hdr = gtt.split("\n")[[i for i, l in enumerate(gtt.split("\n")) if "Balance" in l and ("Date" in l)][0]].lower()
        kws = [k for k in KW if k in hdr]
        low = [t.lower() for t in toks]
        r["kw_recall"] = sum(any(k == t or k in t for t in low) for k in kws) / len(kws)
        rows, cols, _, extra = parse_page_table(lines) if lines else ([], None, None, None)
        rate, checked = reconcile(rows, extra["opening"] if extra else None, 1.0) if rows else (None, 0)
        n_gt = len(gt["rows"])
        r["rows_found"] = len(rows) / n_gt
        r["reconcile_ok_of_gt"] = (sum(x.get("reconciles") == "ok" for x in rows) / max(n_gt - (gt["opening"] is None), 1))
        r["layout"] = gt["layout"]
    else:
        vals = list(gt["fields"].values()); nt = norm(text)
        r["field_exact"] = sum(v in nt for v in vals) / len(vals)
        r["layout"] = "form"
    return r


def main():
    per_page = []
    for f in sorted(glob.glob("results/raw/*.json")):
        R = json.load(open(f))
        for key, pg in R["pages"].items():
            lv, name = key.split("/")
            s = score_page(GT[name], pg["words"])
            per_page.append({"engine": R["engine"], "prep": R["prep"], "level": lv, "page": name,
                             "sec": round(pg["sec"], 2), "peak_rss_mb": round(R["peak_rss_mb"]),
                             "err": pg["err"] or "", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})
    os.makedirs("results", exist_ok=True)
    keys = ["engine", "prep", "level", "page", "layout", "sec", "peak_rss_mb", "cer", "wer", "amount_recall", "date_recall",
            "kw_recall", "rows_found", "reconcile_ok_of_gt", "field_exact", "err"]
    with open("results/results_per_page.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(per_page)
    # summaries
    import pandas as pd
    df = pd.DataFrame(per_page)
    m = ["sec", "cer", "wer", "amount_recall", "date_recall", "kw_recall", "rows_found", "reconcile_ok_of_gt", "field_exact"]
    df["reconcile_AB"] = df["reconcile_ok_of_gt"].where(df["layout"].isin(["A_wd_dep", "B_debit_credit"]))
    m.append("reconcile_AB")
    g = df.groupby(["engine", "prep", "level"])[m].mean().round(3)
    g.to_csv("results/summary_by_level.csv")
    o = df.groupby(["engine", "prep"]).agg({**{k: "mean" for k in m}, "peak_rss_mb": "max"}).round(3)
    o.to_csv("results/summary_overall.csv")
    lay = df[df.layout != "form"].groupby(["engine", "prep", "layout"])[["amount_recall", "rows_found", "reconcile_ok_of_gt"]].mean().round(3)
    lay.to_csv("results/summary_by_layout.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(o.sort_values("amount_recall", ascending=False))


if __name__ == "__main__":
    main()
