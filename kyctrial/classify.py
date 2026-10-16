"""Step 4: label each page bank_statement / application_form / other.

Rule-based scorer first (works with zero labelled data). Once you have labelled
100-200 pages in a CSV (columns: text,label) you can train the optional
TF-IDF + LogisticRegression model and compare."""
import re
import numpy as np
from rapidfuzz import fuzz

from .ocr import group_lines, line_text
from .parsing import DATE_RE, is_amount

STATEMENT_HEADERS = ["credit", "debit", "balance", "withdrawal", "withdrawals", "deposit",
                     "deposits", "cr", "dr", "particulars", "narration", "description",
                     "chq", "cheque", "value date", "txn date", "transaction"]
STATEMENT_TITLE = ["statement of account", "account statement", "bank statement",
                   "opening balance", "closing balance", "ifsc", "account no"]
FORM_WORDS = ["application", "applicant", "co-applicant", "co applicant", "loan",
              "date of birth", "pan", "address", "mobile", "signature", "kyc",
              "occupation", "income", "father", "declaration"]


def _kw_hits(text, keywords):
    t = text.lower()
    return sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", t))


def header_line(lines):
    """Find the most 'table header'-like line: >=2 of credit/debit/balance/withdrawal/deposit.
    Returns (index, line) or (None, None)."""
    core = ["credit", "debit", "balance", "withdrawal", "deposit", "cr", "dr"]
    best, best_i, best_n = None, None, 0
    for i, ln in enumerate(lines):
        toks = [w["text"].lower().strip(".:()") for w in ln]
        n = 0
        for c in core:
            if any(tok == c or tok.startswith(c) or fuzz.ratio(tok, c) >= 85 for tok in toks):
                n += 1
        if n >= 2 and n > best_n:
            best, best_i, best_n = ln, i, n
    return best_i, best


def column_alignment(lines):
    """Fraction of lines whose right edges of amount-like tokens line up with
    other lines (tables have right-aligned numeric columns)."""
    edges = [round(w["x1"] / 15) for ln in lines for w in ln if is_amount(w["text"])]
    if len(edges) < 6:
        return 0.0
    _, counts = np.unique(edges, return_counts=True)
    top = np.sort(counts)[::-1][:3].sum()  # tokens on the 3 busiest right edges
    return float(top / len(edges))


def page_features(words):
    lines = group_lines(words)
    text = "\n".join(line_text(l) for l in lines)
    n_lines = max(len(lines), 1)
    dates = len(DATE_RE.findall(text))
    amounts = sum(1 for w in words if is_amount(w["text"]))
    hi, _ = header_line(lines)
    return {
        "text": text, "lines": lines,
        "header_hits": _kw_hits(text, STATEMENT_HEADERS),
        "title_hits": _kw_hits(text, STATEMENT_TITLE),
        "form_hits": _kw_hits(text, FORM_WORDS),
        "date_density": dates / n_lines,
        "amount_density": amounts / n_lines,
        "has_header_row": hi is not None,
        "alignment": column_alignment(lines),
        "n_words": len(words),
    }


def score_page(f):
    """Weighted evidence -> scores in 0..1. Weights are hand-set starting points;
    tune them on your labelled pages."""
    st = (0.30 * f["has_header_row"]
          + 0.15 * min(f["header_hits"] / 4, 1)
          + 0.10 * min(f["title_hits"] / 2, 1)
          + 0.15 * min(f["date_density"] / 0.4, 1)
          + 0.15 * min(f["amount_density"] / 0.8, 1)
          + 0.15 * f["alignment"])
    form = 0.7 * min(f["form_hits"] / 5, 1) + 0.3 * (1 - min(f["amount_density"] / 0.8, 1))
    if f["n_words"] < 10:
        form = 0.0
    return round(st, 3), round(form, 3)


def classify_page(words, cfg):
    f = page_features(words)
    st, form = score_page(f)
    c = cfg["classify"]
    if st >= c["statement_min_score"] and st >= form:
        label = "bank_statement"
    elif form >= c["form_min_score"]:
        label = "application_form"
    else:
        label = "other"
    return {"label": label, "statement_score": st, "form_score": form,
            "features": {k: v for k, v in f.items() if k not in ("text", "lines")}}, f


# ---------------- optional ML classifier ----------------
def train_ml(labels_csv, model_path):
    """labels_csv columns: text,label (export page texts from outputs/*/pages.csv,
    fix the labels by hand, then train)."""
    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    df = pd.read_csv(labels_csv).dropna(subset=["text", "label"])
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))
    k = min(5, df["label"].value_counts().min())
    if k >= 2:
        print("CV accuracy:", cross_val_score(pipe, df["text"], df["label"], cv=k).mean().round(3))
    pipe.fit(df["text"], df["label"])
    joblib.dump(pipe, model_path)
    return pipe


def predict_ml(model, text):
    proba = model.predict_proba([text])[0]
    i = int(np.argmax(proba))
    return model.classes_[i], float(proba[i])
