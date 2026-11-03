"""Helper functions for main.ipynb. Plain functions only, no classes.
One venv, Python 3.11 or 3.12. See README.md."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- render / preprocess ----------------
def render_pages(pdf_path, dpi=200, pages=None):
    """Render PDF pages to RGB numpy arrays."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    idx = range(len(doc)) if pages is None else pages
    return {i: doc[i].render(scale=dpi / 72).to_numpy()[:, :, :3].copy() for i in idx}

def to_gray(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

def deskew(gray, max_angle=5.0):
    """Small-angle deskew using minAreaRect of dark pixels. Returns (image, angle)."""
    import cv2
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    pts = np.column_stack(np.where(th > 0))
    if len(pts) < 100:
        return gray, 0.0
    ang = cv2.minAreaRect(pts[:, ::-1].astype(np.float32))[-1]
    if ang > 45: ang -= 90
    if abs(ang) < 0.1 or abs(ang) > max_angle:
        return gray, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), float(ang)

# ---------------- OCR (RapidOCR) ----------------
_RAPID = None
def ocr_page(gray, params=None):
    """RapidOCR -> list of {text, score, box[x0,y0,x1,y1]} sorted top-to-bottom."""
    global _RAPID
    if _RAPID is None:
        from rapidocr import RapidOCR
        _RAPID = RapidOCR(params=params or {})
    r = _RAPID(gray)
    out = []
    if r.boxes is not None:
        for b, t, s in zip(r.boxes, r.txts, r.scores):
            b = np.array(b)
            out.append({"text": t, "score": round(float(s), 3),
                        "box": [int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 0].max()), int(b[:, 1].max())]})
    return sorted(out, key=lambda d: (d["box"][1], d["box"][0]))

def lines_from_words(words, y_tol=12):
    """Group OCR boxes into text lines (same row) -> list of strings."""
    rows = []
    for w in sorted(words, key=lambda d: (d["box"][1] + d["box"][3]) / 2):
        yc = (w["box"][1] + w["box"][3]) / 2
        if rows and abs(rows[-1]["y"] - yc) <= y_tol:
            rows[-1]["items"].append(w)
        else:
            rows.append({"y": yc, "items": [w]})
    return [" ".join(i["text"] for i in sorted(r["items"], key=lambda d: d["box"][0])) for r in rows]

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# ---------------- keyword classification ----------------
DATE_RE = re.compile(r"\b(\d{1,2}[-/.\s](\d{1,2}|[A-Za-z]{3,9})[-/.\s]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
AMT_RE = re.compile(r"\b\d{1,3}(,\d{2,3})*\.\d{2}\b|\b\d+\.\d{2}\b")

def keyword_classify(text, keywords, min_words=10):
    """Score page text against each class word list; returns (label, scores).
    keywords = {"statement": [...], "applicant": [...]} ; anything weak -> not_needed."""
    t = text.lower()
    n_words = max(len(t.split()), 1)
    scores = {}
    for cls, words in keywords.items():
        hits = sum(1 for w in words if w.lower() in t)
        scores[cls] = round(min(hits / max(len(words) * 0.3, 1), 1.0), 3)
    # numeric signals help statements (many dates + amounts per page)
    dens = (len(DATE_RE.findall(text)) + len(AMT_RE.findall(text))) / n_words
    if "statement" in scores:
        scores["statement"] = round(0.7 * scores["statement"] + 0.3 * min(dens / 0.3, 1.0), 3)
    if n_words < min_words:
        return "not_needed", scores
    best = max(scores, key=scores.get)
    return (best if scores[best] >= 0.35 else "not_needed"), scores

# ---------------- BERT (optional, local weights only) ----------------
def bert_embed(texts, model_dir, max_len=256):
    """Mean-pooled embeddings from a local BERT-family folder (no HF hub calls)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        e = (h * m).sum(1) / m.sum(1)
    return torch.nn.functional.normalize(e, dim=1).numpy()

def bert_classify(page_texts, prototypes, model_dir, min_sim=0.0):
    """Nearest-prototype classification. prototypes = {label: [example text, ...]}.
    Untrained BERT => weak; replace prototypes with real labelled pages / fine-tune later."""
    labels = list(prototypes)
    proto = [np.mean(bert_embed(prototypes[l], model_dir), axis=0) for l in labels]
    proto = np.stack([p / np.linalg.norm(p) for p in proto])
    sims = bert_embed(page_texts, model_dir) @ proto.T
    out = []
    for row in sims:
        j = int(row.argmax())
        out.append((labels[j] if row[j] >= min_sim else "not_needed", {l: round(float(s), 3) for l, s in zip(labels, row)}))
    return out

# ---------------- applicant / account fields ----------------
FIELD_LABELS = {
    "customer_id": ["customer id", "cust id", "customer no", "cif", "client id"],
    "account_no": ["account number", "account no", "a/c no", "acct no"],
    "account_name": ["account holder", "account name", "customer name"],
    "opening_balance": ["opening balance", "balance b/f", "brought forward", "opening bal"],
    "closing_balance": ["closing balance", "balance c/f", "carried forward", "closing bal"],
    "period": ["statement period", "period"],
}

def parse_applicant_fields(lines, labels=FIELD_LABELS, cutoff=85):
    """For each field, find the line whose start fuzzy-matches a label; take text after it."""
    from rapidfuzz import fuzz
    out = {}
    for field, labs in labels.items():
        best = (0, None)
        for ln in lines:
            low = ln.lower()
            for lab in labs:
                s = fuzz.partial_ratio(lab, low)
                if s >= cutoff and s > best[0]:
                    pos = low.find(lab)
                    val = ln[pos + len(lab):] if pos >= 0 else ln.split(":", 1)[-1]
                    val = val.lstrip(" :.-|").strip()
                    if val:
                        best = (s, val)
        out[field] = best[1]
    for k in ("opening_balance", "closing_balance"):
        if out.get(k):
            m = AMT_RE.search(out[k].replace(" ", ""))
            out[k] = float(m.group().replace(",", "")) if m else out[k]
    return out

# ---------------- PP-StructureV3 (run in .venv-paddle) ----------------
_PPS = None
def ppstructure_pipeline(**kw):
    global _PPS
    if _PPS is None:
        import os
        # PaddleX defaults to Hugging Face. Force Baidu BOS and block the hub.
        os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["HF_HUB_OFFLINE"] = "1"
        from paddleocr import PPStructureV3
        cfg = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                   use_textline_orientation=False, use_formula_recognition=False,
                   use_chart_recognition=False, use_seal_recognition=False,
                   use_table_recognition=True, device="cpu",
                   enable_mkldnn=False,  # mkldnn leaked RAM (OOM at ~12 GB on 4 pages)
                   text_detection_model_name="PP-OCRv5_mobile_det",
                   text_recognition_model_name="PP-OCRv5_mobile_rec")
        cfg.update(kw)
        _PPS = PPStructureV3(**cfg)
    return _PPS

def ppstructure_run(img_bgr, **kw):
    """Run PP-StructureV3 on one page image -> result json dict ('res')."""
    res = next(iter(ppstructure_pipeline(**kw).predict(img_bgr)))
    return res.json["res"]

def table_html_to_df(html):
    from io import StringIO
    return pd.read_html(StringIO(html))[0]

# ---------------- normalise / months / outputs ----------------
COL_MAP = {
    "date": ["value date", "txn date", "transaction date", "date", "posted"],
    "description": ["description", "narration", "particulars", "details", "remarks"],
    "drcr": ["cr/dr", "dr/cr", "type"],
    "debit": ["debit", "withdrawal", "dr"],
    "credit": ["credit", "deposit", "cr"],
    "amount": ["amount"],
    "balance": ["balance", "available balance"],
}

def _parse_dates(col):
    s = col.astype(str).str.extract(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s?[A-Za-z]{3}\s?\d{2,4})")[0]
    return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")

def _to_num(col):
    return pd.to_numeric(col.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")

def normalize_columns(df, col_map=COL_MAP, cutoff=80):
    """Fuzzy-map raw headers to standard names. Header may arrive as row 0 (PP-StructureV3 html).
    Falls back to content checks when headers are shifted/merged (common on noisy scans).
    Returns empty frame if the table has no date-like column (e.g. account-summary tables)."""
    from rapidfuzz import fuzz
    df = df.copy()
    if all(isinstance(c, int) for c in df.columns):
        df.columns = [str(x) for x in df.iloc[0]]; df = df.iloc[1:]
    df.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    df = df.loc[:, ~df.isna().all()]                                   # drop empty columns
    df.columns = [f"{c}_{i}" if list(df.columns).count(c) > 1 else c for i, c in enumerate(df.columns)]
    used, rename = set(), {}
    for std, labs in col_map.items():
        best = None
        for c in df.columns:
            if c in used: continue
            sc = max(max(fuzz.ratio(c.lower(), l), fuzz.token_set_ratio(c.lower(), l) - 5) for l in labs)
            if sc >= cutoff and (best is None or sc > best[1]):
                best = (c, sc)
        if best:
            rename[best[0]] = std; used.add(best[0])
    df = df.rename(columns=rename)
    # content fallbacks
    if "date" not in df or _parse_dates(df["date"]).notna().mean() < 0.5:
        rates = {c: _parse_dates(df[c]).notna().mean() for c in df.columns}
        c = max(rates, key=rates.get) if rates else None
        if not c or rates[c] < 0.5:
            return pd.DataFrame()
        df = df.drop(columns=[x for x in ["date"] if x in df and x != c]).rename(columns={c: "date"})
    df["date"] = _parse_dates(df["date"])
    if "drcr" not in df:
        for c in df.columns:
            ex = df[c].astype(str).str.extract(r"^\s*(cr|dr)\b\s*(.*)$", flags=2)
            if ex[0].notna().mean() > 0.5:
                df["drcr"] = ex[0]; df["amount"] = ex[1]; break
    if "drcr" in df: df["drcr"] = df["drcr"].astype(str).str.strip().str.upper()
    for c in ("amount", "debit", "credit", "balance"):
        if c in df: df[c] = _to_num(df[c])
    return df[df["date"].notna()].reset_index(drop=True)

def months_covered(dates):
    d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    return sorted(d.dt.to_period("M").astype(str).unique())

def banking_flags(months_available, full_months=6):
    return {"no_banking_data": int(months_available == 0),
            "partial_banking_data": int(0 < months_available < full_months),
            "full_banking_data": int(months_available >= full_months),
            "months_available": int(months_available)}

def write_outputs(lead_row, transactions, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    transactions.to_excel(out / "transactions.xlsx", index=False)
    pd.DataFrame([lead_row]).to_excel(out / "leads.xlsx", index=False)
    return out / "transactions.xlsx", out / "leads.xlsx"

# ================= Stage 1-2-3 recall-first page classification =================
# Labels: applicant | statement | not_needed. Policy: KEEP on any positive signal;
# not_needed only when every stage agrees. Text always from RapidOCR (page_text).
FIN_HEADERS = ["date", "value date", "txn date", "narration", "description", "particulars", "debit",
               "credit", "withdrawal", "deposit", "balance", "closing balance", "opening balance",
               "cheque", "chq no", "ref no", "dr", "cr", "amount", "transaction", "statement of account"]
APPLICANT_CUES = ["application", "applicant", "loan", "pan", "aadhaar", "date of birth", "dob",
                  "father", "address", "mobile", "email", "occupation", "kyc", "customer id",
                  "signature", "photograph", "nominee", "annual income", "employer"]
APP_ID_RE = re.compile(r"\b(app(lication)?\s*(id|no|number)|lead\s*id)\s*[:#.\-]?\s*[A-Z0-9\-/]{5,}", re.I)
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
AADHAAR_RE = re.compile(r"\b(?:[Xx*]{4}[\s-]?){2}\d{4}\b|\b\d{4}\s\d{4}\s\d{4}\b")

CLASS_DESC = {
    "statement": "a bank account statement page listing transactions with dates, amounts and balances",
    "applicant": "a loan application or KYC form with applicant personal details, ID numbers and address",
    "not_needed": "a terms and conditions, cover, blank, advertisement or other irrelevant page",
}

def stage1_rules(text, keywords=None, fuzz_cutoff=88):
    """Cheap rules. Returns (label|None, signals). None = no positive -> go to stage 2."""
    from rapidfuzz import fuzz
    lines = [l.lower() for l in text.splitlines() if l.strip()]
    t = "\n".join(lines)
    rows = sum(1 for l in lines if DATE_RE.search(l) and AMT_RE.search(l))
    hdr = sum(1 for h in FIN_HEADERS if any(fuzz.partial_ratio(h, l) >= fuzz_cutoff for l in lines[:80])) if lines else 0
    app_hits = sum(1 for c in APPLICANT_CUES if c in t)
    ids = bool(APP_ID_RE.search(text)) + bool(PAN_RE.search(text)) + bool(AADHAAR_RE.search(text))
    kw = keyword_classify(text, keywords)[0] if keywords else "not_needed"
    s = {"txn_rows": rows, "fin_headers": hdr, "app_cues": app_hits, "id_hits": ids, "kw": kw}
    stmt = rows >= 3 or hdr >= 4 or kw == "statement"
    app = ids >= 1 or app_hits >= 3 or kw == "applicant"
    if stmt and app:   # both -> keep, pick the stronger
        return ("statement" if rows >= 3 else "applicant"), s
    if stmt: return "statement", s
    if app: return "applicant", s
    return None, s

_CACHE = {}
def _layout(name):
    if name not in _CACHE:
        from paddleocr import LayoutDetection
        _CACHE[name] = LayoutDetection(model_name=name)
    return _CACHE[name]

def layout_signals(img_rgb, model="PP-DocLayout-S", min_score=0.4):
    bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
    r = list(_layout(model).predict(bgr))[0].json["res"]
    labs = [b["label"] for b in r["boxes"] if b["score"] >= min_score]
    return {k: labs.count(k) for k in set(labs)}

def _zs(model):
    if model not in _CACHE:
        from transformers import pipeline
        _CACHE[model] = pipeline("zero-shot-classification", model=model, device=-1)
    return _CACHE[model]

def zeroshot_scores(text, model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33", max_chars=1500):
    r = _zs(model)(text[:max_chars] or "empty page", list(CLASS_DESC.values()), multi_label=True,
                    hypothesis_template="This is {}.")
    inv = {v: k for k, v in CLASS_DESC.items()}
    return {inv[l]: round(float(s), 3) for l, s in zip(r["labels"], r["scores"])}

def rvl_scores(img_rgb, model="jordyvl/vit-tiny_rvl_cdip", proc="WinKawaks/vit-tiny-patch16-224"):
    """ViT-Tiny RVL-CDIP (16 doc types). Returns top label dict."""
    if model not in _CACHE:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        _CACHE[model] = (AutoImageProcessor.from_pretrained(proc), AutoModelForImageClassification.from_pretrained(model).eval())
    import torch
    from PIL import Image
    p, m = _CACHE[model]
    with torch.no_grad():
        pr = m(**p(Image.fromarray(img_rgb).convert("RGB"), return_tensors="pt")).logits.softmax(-1)[0]
    top = pr.topk(3)
    return {m.config.id2label[int(i)]: round(float(v), 3) for v, i in zip(top.values, top.indices)}

RVL_KEEP = {"form", "invoice", "budget", "questionnaire", "specification", "resume", "memo", "letter"}

def classify_page(text, img_rgb, keywords=None, zs_keep=0.30, s3_keep=0.20):
    """Recall-first cascade. Returns (label, stage, info)."""
    lab, s1 = stage1_rules(text, keywords)
    info = {"s1": s1}
    if lab: return lab, "S1_rules", info
    # Stage 2: PP-DocLayout-S + DeBERTa-xsmall; OR-keep
    try:
        lay = layout_signals(img_rgb, "PP-DocLayout-S"); zs = zeroshot_scores(text)
    except Exception as e:  # models unavailable (e.g. HF_HUB_OFFLINE, no torch) -> recall-first keep
        info["error"] = str(e)[:120]
        return ("applicant" if s1["app_cues"] >= 1 else "statement"), "S2_unavailable_keep", info
    info.update(s2_layout=lay, s2_zs=zs)
    if lay.get("table", 0) >= 1 or zs["statement"] >= zs_keep:
        return "statement", "S2_layout/zs", info
    if zs["applicant"] >= zs_keep or (lay.get("seal", 0) and s1["app_cues"] >= 1):
        return "applicant", "S2_layout/zs", info
    # Stage 3: only to CONFIRM not_needed; any doubt keeps
    try:
        lay3 = layout_signals(img_rgb, "PP-DocLayout-M")
        zs3 = zeroshot_scores(text, "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c")
    except Exception as e:
        info["error"] = str(e)[:120]
        return ("applicant" if zs["applicant"] >= zs["statement"] else "statement"), "S3_unavailable_keep", info
    try: rvl = rvl_scores(img_rgb)
    except Exception as e: rvl = {"error": str(e)[:80]}
    info.update(s3_layout=lay3, s3_zs=zs3, s3_rvl=rvl)
    if lay3.get("table", 0) >= 1 or zs3["statement"] >= s3_keep:
        return "statement", "S3_doubt_keep", info
    rvl_top = next(iter(rvl), "")
    if zs3["applicant"] >= s3_keep or rvl_top in RVL_KEEP or "error" in rvl:
        return "applicant", "S3_doubt_keep", info
    return "not_needed", "S3_confirmed_drop", info

def smooth_promote(labels):
    """Promote-only: a not_needed page sandwiched between kept pages takes the previous kept label."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        if out[i] == "not_needed" and labels[i-1] != "not_needed" and labels[i+1] != "not_needed":
            out[i] = labels[i-1]
    return out
