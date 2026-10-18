"""Everything except OCR lives here, top to bottom in pipeline order:

    settings  ->  dates/amounts  ->  render  ->  deskew  ->  OCR cache  ->  lines
    ->  classify  ->  statement table + balance check + months  ->  fields
    ->  process_pdf / run_folder  ->  evaluate

Settings are plain constants right below. Edit them and re-run (in the notebook:
`importlib.reload(common)`). Every word everywhere is a dict:
    {"text": str, "x0": float, "y0": float, "x1": float, "y1": float, "conf": float}
in pixel coordinates at DPI (text-layer words are scaled to match OCR output)."""
import csv
import glob
import hashlib
import json
import os
import re
import time
from datetime import date

import cv2
import numpy as np
import pdfplumber
import pypdfium2 as pdfium
from dateutil import parser as dparser
from rapidfuzz import fuzz

# =============================== SETTINGS ===================================
DPI = 250               # render DPI for scanned pages (200-300 is the sweet spot)
MIN_TEXT_CHARS = 80     # page text layer with fewer chars -> treat as scanned, OCR it
DESKEW = True           # straighten small scanner rotations
MAX_SKEW_DEG = 10       # bigger estimated angles are ignored (probably wrong)
BINARISE = False        # adaptive threshold to pure B/W. OFF: hurt RapidOCR badly (bench/)
CACHE_DIR = "outputs/_ocr_cache"   # OCR JSON per page; set None to disable. Delete after trial!

STATEMENT_MIN_SCORE = 0.45   # page is a bank statement if st >= this (and st >= form)
FORM_MIN_SCORE = 0.35        # else application form if form >= this
BALANCE_TOL = 1.0            # rupees, for prev + credit - debit == balance
MAX_PAGE_GAP = 1             # statement pages separated by <= this many other pages = one statement
FUZZY_THRESHOLD = 85         # label fuzzy match (0-100)

STATEMENT_HEADERS = ["credit", "debit", "balance", "withdrawal", "withdrawals", "deposit",
                     "deposits", "cr", "dr", "particulars", "narration", "description",
                     "chq", "cheque", "value date", "txn date", "transaction"]
STATEMENT_TITLE = ["statement of account", "account statement", "bank statement",
                   "opening balance", "closing balance", "ifsc", "account no"]
FORM_WORDS = ["application", "applicant", "co-applicant", "co applicant", "loan",
              "date of birth", "pan", "address", "mobile", "signature", "kyc",
              "occupation", "income", "father", "declaration"]
HEADER_CORE = ["credit", "debit", "balance", "withdrawal", "deposit", "cr", "dr"]  # >=2 on a line = table header

COLUMN_SYNONYMS = {  # add the client's bank header names here
    "date": ["date", "txn date", "tran date", "transaction date", "value date", "post date"],
    "description": ["description", "particulars", "narration", "details", "remarks", "transaction details"],
    "ref": ["chq", "cheque", "ref", "chq/ref", "reference", "instrument"],
    "debit": ["debit", "withdrawal", "withdrawals", "dr", "withdrawal amt", "debit amount"],
    "credit": ["credit", "deposit", "deposits", "cr", "deposit amt", "credit amount"],
    "balance": ["balance", "closing balance", "bal"],
}
STOP_WORDS = ("closing balance", "total", "grand total", "end of statement", "page total")

ID_RX = r"^(?=.*\d)[A-Z0-9][A-Z0-9/\-]{5,24}$"
NAME_RX = r"^[A-Za-z][A-Za-z .']{2,60}$"
FIELDS = {  # label synonyms + validator regex per field. Add the client's real labels.
    "app_id": {"labels": ["application id", "application no", "application number", "app id", "app no",
                          "loan application no", "lan", "reference no", "ref no"], "regex": ID_RX},
    "name": {"labels": ["applicant name", "name of applicant", "name of the applicant", "customer name",
                        "borrower name", "name"], "regex": NAME_RX},
    "co_app_name": {"labels": ["co-applicant name", "co applicant name", "name of co-applicant",
                               "co-borrower name", "co borrower name"], "regex": NAME_RX},
    "co_app_id": {"labels": ["co-applicant id", "co applicant id", "co-applicant application id",
                             "co-applicant application no", "co-borrower id"], "regex": ID_RX},
}
# a label for field X must not fire inside a line that belongs to field Y
FIELD_EXCLUDE = {
    "name": ["co-app", "co app", "coapp", "co-borrower", "co borrower", "father", "spouse",
             "mother", "bank", "branch", "nominee", "account name"],
    "app_id": ["co-app", "co app", "coapp", "co-borrower"],
}
# words that mean we grabbed another label instead of a value
NOT_A_VALUE = {"date", "birth", "name", "applicant", "application", "pan", "mobile", "address",
               "father", "loan", "amount", "occupation", "id", "no", "number", "signature"}

# ============================ DATES & AMOUNTS ===============================
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
# dd/mm/yyyy, dd-mm-yy, dd.mm.yyyy, dd-Mon-yyyy, dd Mon yyyy, dd-Mon-yy
DATE_RE = re.compile(
    rf"\b(\d{{1,2}}[/\-.]\d{{1,2}}[/\-.](?:\d{{4}}|\d{{2}})"
    rf"|\d{{1,2}}[\-\s/]?(?:{MONTHS})[a-z]*[\-\s/,]*(?:\d{{4}}|\d{{2}}))\b", re.I)
# 1,23,456.78 | 123456.78 | 1,234.00 Cr | 500.00Dr  (needs 2 decimals to avoid IDs/years)
AMOUNT_RE = re.compile(r"^-?\(?[₹]?\s?\d{1,3}(?:,\d{2,3})*(?:\.\d{2})\)?\s?(?:cr|dr)?\.?$"
                       r"|^-?\d+\.\d{2}\s?(?:cr|dr)?\.?$", re.I)
_D = DATE_RE.pattern.replace("\\b(", "\\b(?:", 1)
PERIOD_RE = re.compile(
    rf"(?:period|from|statement\s+for)\b[^0-9]{{0,40}}(?P<a>{_D})\s*(?:to|till|-|–)\s*(?P<b>{_D})", re.I)
_LENIENT_DATE = re.compile(r"^(\d{1,2})[^\d]?(\d{2})(?:[^\d]|1)?(\d{4}|\d{2})$")


def _ok_year(d):
    return 1990 <= d.year <= date.today().year + 1


def parse_date(s):
    """Indian-style date (day first) anywhere in s -> datetime.date or None."""
    m = DATE_RE.search(s.strip().replace(",", " "))
    if not m:
        return None
    try:
        d = dparser.parse(m.group(0).replace(".", "/"), dayfirst=True, fuzzy=True,
                          default=dparser.parse("2000-01-01")).date()
    except (ValueError, OverflowError):
        return None
    return d if _ok_year(d) else None


def parse_date_lenient(token):
    """OCR-tolerant date for the FIRST cell of a table row only:
    '02:01:24', '260124', '24/01124' (slash read as 1) -> date."""
    t = token.strip(" .:;,|")
    d = parse_date(t)
    if d:
        return d
    m = _LENIENT_DATE.match(t)
    if not m:
        return None
    dd, mm, yy = (int(x) for x in m.groups())
    yy += 2000 if yy < 100 else 0
    try:
        d = date(yy, mm, dd)
    except ValueError:
        return None
    return d if _ok_year(d) else None


def normalise_amount_text(s):
    """Fix common OCR slips: '117.886.29' -> '117,886.29', 'C1'/'Ct' -> 'cr',
    '$'->'5', 'O'->'0'. Extend as you see new errors in your scans."""
    t = s.strip().replace(" ", "").replace("₹", "").strip(":;'`|!+").replace("'", "")
    if sum(ch.isdigit() for ch in t) >= max(3, len(t) // 2):  # mostly digits -> fix letter confusions
        t = t.translate(str.maketrans({"$": "5", "S": "5", "s": "5", "O": "0", "o": "0",
                                       "l": "1", "I": "1", ":": ".", ";": "."}))
        t = t.strip(".") if not t.lower().endswith(("cr", "dr")) else t
    t = re.sub(r"(?i)(c[r1t7l]|d[r1])\.?:?$", lambda m: "cr" if m.group(0)[0] in "cC" else "dr", t)
    if t.count(".") > 1:  # dots used as thousand separators by OCR
        head, tail = t.rsplit(".", 1)
        t = head.replace(".", ",") + "." + tail
    if t.endswith(".") and not t.lower().endswith(("cr.", "dr.")):
        t = t[:-1]
    return t.lstrip(".,")


def is_amount(s):
    return bool(AMOUNT_RE.match(normalise_amount_text(s)))


def parse_amount(s):
    """'1,23,456.78 Cr' -> (123456.78, 'cr'); not an amount -> (None, None)."""
    t = normalise_amount_text(s)
    if not AMOUNT_RE.match(t):
        return None, None
    sign, low = None, t.lower().rstrip(".")
    if low.endswith(("cr", "dr")):
        sign, low = low[-2:], low[:-2]
    neg = low.startswith(("-", "("))
    val = float(low.strip("-()").replace(",", ""))
    return (-val if neg else val), sign

# ================================ RENDER ====================================
def page_count(pdf_path):
    pdf = pdfium.PdfDocument(pdf_path)
    n = len(pdf)
    pdf.close()
    return n


def render_page(pdf_path, page_no, dpi=DPI):
    """Page (0-based) -> RGB numpy image."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        img = pdf[page_no].render(scale=dpi / 72.0).to_numpy()
    finally:
        pdf.close()
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def text_layer_words(pdf_path, page_no, dpi=DPI):
    """Words from the embedded text layer (digital PDFs), scaled to pixel space."""
    s = dpi / 72.0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            raw = pdf.pages[page_no].extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1.5)
    except Exception:  # broken/odd text layer -> OCR instead
        return []
    return [{"text": w["text"], "x0": w["x0"] * s, "y0": w["top"] * s,
             "x1": w["x1"] * s, "y1": w["bottom"] * s, "conf": 1.0} for w in raw]

# ================================ DESKEW ====================================
def estimate_skew(gray, max_deg=MAX_SKEW_DEG):
    """Median angle of long thin text-line blobs (minAreaRect on smeared dark pixels)."""
    _, bw = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    smear = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3)))
    contours, _ = cv2.findContours(smear, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        _, (w, h), a = cv2.minAreaRect(c)
        if w < h:
            w, h, a = h, w, a - 90
        if w > 150 and h < 60:
            a = a - 90 if a > 45 else a + 90 if a < -45 else a
            angles.append(a)
    ang = float(np.median(angles)) if angles else 0.0
    return ang if abs(ang) <= max_deg else 0.0


def preprocess(img):
    """Grayscale -> deskew (-> optional binarise). Returns (gray image, angle)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    angle = 0.0
    if DESKEW:
        # estimate on a median-filtered copy: salt & pepper specks otherwise break the line blobs
        angle = estimate_skew(cv2.medianBlur(gray, 3))
        if abs(angle) > 0.1:
            h, w = gray.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if BINARISE:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return gray, angle

# ============================ PAGE -> WORDS =================================
def _cache_path(pdf_path, page_no, engine):
    st = os.stat(pdf_path)
    raw = f"{os.path.abspath(pdf_path)}|{st.st_size}|{st.st_mtime}|{page_no}|{DPI}|{DESKEW}|{BINARISE}|{engine}"
    return os.path.join(CACHE_DIR, hashlib.sha1(raw.encode()).hexdigest()[:16] + ".json")


def get_page_words(pdf_path, page_no, ocr_page, engine="rapid"):
    """One page -> dict(source='text'|'ocr', words, image, angle, cached).
    Uses the text layer if present, else deskew + OCR (with the JSON cache)."""
    image = render_page(pdf_path, page_no)
    words = text_layer_words(pdf_path, page_no)
    if sum(len(w["text"]) for w in words) >= MIN_TEXT_CHARS:
        return {"source": "text", "words": words, "image": image, "angle": 0.0, "cached": None}
    clean, angle = preprocess(image)
    path = _cache_path(pdf_path, page_no, engine) if CACHE_DIR else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {"source": "ocr", "words": json.load(f), "image": clean, "angle": angle, "cached": True}
    words = ocr_page(clean)
    if path:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(words, f)
    return {"source": "ocr", "words": words, "image": clean, "angle": angle, "cached": False}


def group_lines(words, y_tol=0.5):
    """Words -> text lines (vertical overlap), top-to-bottom, each left-to-right."""
    lines = []
    for w in sorted(words, key=lambda w: (w["y0"] + w["y1"]) / 2):
        cy, h = (w["y0"] + w["y1"]) / 2, max(w["y1"] - w["y0"], 1)
        for ln in lines:
            if abs(ln["cy"] - cy) < y_tol * max(h, ln["h"]):
                ln["words"].append(w)
                n = len(ln["words"])
                ln["cy"] = (ln["cy"] * (n - 1) + cy) / n
                break
        else:
            lines.append({"cy": cy, "h": h, "words": [w]})
    lines.sort(key=lambda l: l["cy"])
    return [sorted(l["words"], key=lambda w: w["x0"]) for l in lines]


def line_text(line):
    return " ".join(w["text"] for w in line)

# =============================== CLASSIFY ===================================
def _kw_hits(text, keywords):
    t = text.lower()
    return sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", t))


def header_line(lines):
    """Most table-header-like line (>=2 of HEADER_CORE, fuzzy). Returns (index, line) or (None, None)."""
    best, best_i, best_n = None, None, 0
    for i, ln in enumerate(lines):
        toks = [w["text"].lower().strip(".:()") for w in ln]
        n = sum(1 for c in HEADER_CORE
                if any(t == c or t.startswith(c) or fuzz.ratio(t, c) >= 85 for t in toks))
        if n >= 2 and n > best_n:
            best, best_i, best_n = ln, i, n
    return best_i, best


def column_alignment(lines):
    """Share of amount tokens sitting on the 3 busiest right edges (tables are right-aligned)."""
    edges = [round(w["x1"] / 15) for ln in lines for w in ln if is_amount(w["text"])]
    if len(edges) < 6:
        return 0.0
    _, counts = np.unique(edges, return_counts=True)
    return float(np.sort(counts)[::-1][:3].sum() / len(edges))


def classify_page(words):
    """Rule-based page label. Returns (result dict, lines).
    st   = how much it looks like a bank statement (0..1)
    form = how much it looks like an application form (0..1)"""
    lines = group_lines(words)
    text = "\n".join(line_text(l) for l in lines)
    n_lines = max(len(lines), 1)
    f = {"header_hits": _kw_hits(text, STATEMENT_HEADERS),
         "title_hits": _kw_hits(text, STATEMENT_TITLE),
         "form_hits": _kw_hits(text, FORM_WORDS),
         "date_density": round(len(DATE_RE.findall(text)) / n_lines, 3),
         "amount_density": round(sum(is_amount(w["text"]) for w in words) / n_lines, 3),
         "has_header_row": header_line(lines)[0] is not None,
         "alignment": round(column_alignment(lines), 3),
         "n_words": len(words)}
    st = (0.30 * f["has_header_row"]
          + 0.15 * min(f["header_hits"] / 4, 1)
          + 0.10 * min(f["title_hits"] / 2, 1)
          + 0.15 * min(f["date_density"] / 0.4, 1)
          + 0.15 * min(f["amount_density"] / 0.8, 1)
          + 0.15 * f["alignment"])
    form = 0.7 * min(f["form_hits"] / 5, 1) + 0.3 * (1 - min(f["amount_density"] / 0.8, 1))
    if f["n_words"] < 10:
        form = 0.0
    st, form = round(st, 3), round(form, 3)
    if st >= STATEMENT_MIN_SCORE and st >= form:
        label = "bank_statement"
    elif form >= FORM_MIN_SCORE:
        label = "application_form"
    else:
        label = "other"
    return {"label": label, "statement_score": st, "form_score": form, "features": f}, lines

# ============================= STATEMENTS ===================================
NUMERIC = ("debit", "credit", "balance")


def group_runs(labels, max_gap=MAX_PAGE_GAP):
    """['other','bank_statement','bank_statement','other','bank_statement'] -> [[1, 2, 4]]"""
    runs, cur, last = [], [], None
    for i, lab in enumerate(labels):
        if lab != "bank_statement":
            continue
        if cur and i - last - 1 > max_gap:
            runs.append(cur)
            cur = []
        cur.append(i)
        last = i
    return runs + [cur] if cur else runs


def _col_type(text):
    t = text.lower().strip(" .:()")
    for col in ("balance", "debit", "credit", "description", "ref", "date"):
        for syn in COLUMN_SYNONYMS[col]:
            if t == syn or t.startswith(syn) or (len(syn) > 3 and syn in t):
                return col
    return None


def header_columns(hline):
    """Merge header words into cells (small gaps), type each cell via COLUMN_SYNONYMS."""
    gap_tol = sum(w["y1"] - w["y0"] for w in hline) / len(hline) * 0.9
    cells = []
    for w in hline:
        if cells and w["x0"] - cells[-1]["x1"] < gap_tol and _col_type(w["text"]) is None:
            cells[-1]["text"] += " " + w["text"]
            cells[-1]["x1"] = w["x1"]
        else:
            cells.append(dict(text=w["text"], x0=w["x0"], x1=w["x1"]))
    cols = []
    for c in cells:
        t = _col_type(c["text"])
        if t and t not in [x["type"] for x in cols]:
            cols.append({**c, "type": t})
    return cols


def assign_column(w, cols):
    """Amounts -> numeric column with nearest RIGHT edge; text -> nearest column centre."""
    if is_amount(w["text"]):
        num = [c for c in cols if c["type"] in NUMERIC]
        if num:
            return min(num, key=lambda c: abs(c["x1"] - w["x1"]))["type"]
    cx = (w["x0"] + w["x1"]) / 2
    return min(cols, key=lambda c: abs((c["x0"] + c["x1"]) / 2 - cx))["type"]


def parse_page_table(lines, cols=None):
    """Statement page lines -> (rows, cols, opening_balance).
    `cols` from the previous page are reused when this page has no header row."""
    hi, hline = header_line(lines)
    start = 0
    if hline:
        new_cols = header_columns(hline)
        if sum(c["type"] in NUMERIC for c in new_cols) >= 2:
            cols, start = new_cols, hi + 1
    if not cols:
        return [], None, None
    rows, opening = [], None
    for ln in lines[start:]:
        txt = line_text(ln).lower()
        if any(s in txt for s in STOP_WORDS):
            if rows:
                break
            continue
        cells = {}
        for w in ln:
            cells.setdefault(assign_column(w, cols), []).append(w["text"])
        d = parse_date(" ".join(cells["date"])) if "date" in cells else None
        if d is None and ln and ln[0]["x0"] < cols[0]["x1"] + 50 and not is_amount(ln[0]["text"]):
            d = parse_date_lenient(ln[0]["text"])  # OCR-damaged date in the first column
        if "opening balance" in txt:
            amts = [parse_amount(w["text"]) for w in ln if is_amount(w["text"])]
            if amts:
                v, s = amts[-1]
                opening = -v if s == "dr" else v
            continue
        if d is not None:
            row = {"date": d.isoformat(), "description": " ".join(cells.get("description", [])),
                   "debit": None, "credit": None, "balance": None, "raw": line_text(ln)}
            for col in NUMERIC:
                vals = [v for v in (parse_amount(t) for t in cells.get(col, [])) if v[0] is not None]
                if vals:
                    v, s = vals[-1]
                    row[col] = -v if (col == "balance" and s == "dr") else v
            rows.append(row)
        elif rows and cells.get("description") and not any(is_amount(w["text"]) for w in ln):
            rows[-1]["description"] += " " + " ".join(cells["description"])  # wrapped narration
    return rows, cols, opening


def balance_check(rows, opening=None, tol=BALANCE_TOL):
    """Sets row['balance_ok'] = ok/fail/na using prev + credit - debit == balance.
    Returns (share ok, rows checked)."""
    prev, ok, checked = opening, 0, 0
    for r in rows:
        r["balance_ok"] = "na"
        if r["balance"] is None:
            continue
        if prev is not None:
            checked += 1
            if abs(prev + (r["credit"] or 0) - (r["debit"] or 0) - r["balance"]) <= tol:
                ok += 1
                r["balance_ok"] = "ok"
            else:
                r["balance_ok"] = "fail"
        prev = r["balance"]
    return (round(ok / checked, 3) if checked else None), checked


def months_between(a, b):
    return (b.year - a.year) * 12 + b.month - a.month + 1


def find_period(text):
    """'Statement period 01/04/2023 to 30/09/2023' -> (date, date) or None."""
    m = PERIOD_RE.search(text.replace("\n", " "))
    if not m:
        return None
    a, b = parse_date(m.group("a")), parse_date(m.group("b"))
    return (a, b) if a and b and a <= b else None


def analyse_statement(run_pages, page_lines):
    """One statement (consecutive pages) -> (summary dict, rows).
    months: from the 'period ... to ...' line if found, else first..last transaction month."""
    cols, rows, opening, period = None, [], None, None
    for p in run_pages:
        lines = page_lines[p]
        period = period or find_period("\n".join(line_text(l) for l in lines[:25]))
        page_rows, cols, op = parse_page_table(lines, cols)
        if opening is None and op is not None and not rows:
            opening = op
        for r in page_rows:
            r["page"] = p + 1
        rows += page_rows
    rate, checked = balance_check(rows, opening)
    dates = [date.fromisoformat(r["date"]) for r in rows]
    s = {"pages": [p + 1 for p in run_pages], "n_rows": len(rows), "balance_ok_rate": rate,
         "rows_checked": checked, "opening_balance": opening,
         "txn_start": min(dates).isoformat() if dates else None,
         "txn_end": max(dates).isoformat() if dates else None,
         "txn_span_months": months_between(min(dates), max(dates)) if dates else 0,
         "distinct_txn_months": len({(d.year, d.month) for d in dates}),
         "period_start": None, "period_end": None, "period_months": None}
    if period:
        s.update(period_start=period[0].isoformat(), period_end=period[1].isoformat(),
                 period_months=months_between(*period))
    s["months"] = s["period_months"] or s["txn_span_months"]
    return s, rows

# ================================ FIELDS ====================================
_JUNK = re.compile(r"^[\s:;\-–_.|]+|[\s:;\-–_.|]+$")


def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def _right_value(line, spans, end_char):
    """Value to the right of the label on the same line (stops at a big gap / next label)."""
    words = [(w, s) for w, s in zip(line, spans) if s[1] > end_char]
    if not words:
        return None
    h = sum(w["y1"] - w["y0"] for w, _ in words) / len(words)
    out, last_x1 = [], None
    for w, (a, _) in words:
        t = w["text"][end_char - a:] if a < end_char else w["text"]  # e.g. "Name:Rahul"
        if not _JUNK.sub("", t):
            continue
        if out and w["x0"] - last_x1 > 3 * h:
            break
        if out and len(t) > 3 and any(fuzz.partial_ratio(l, _norm(t)) >= 95
                                      for l in ("application", "co-applicant", "mobile", "date", "pan")):
            break
        out.append(t)
        last_x1 = w["x1"]
    return _JUNK.sub("", " ".join(out)).strip() or None


def _below_value(lines, i, x0, x1):
    """Value on the next line, horizontally under the label."""
    if i + 1 >= len(lines):
        return None
    ws = [w for w in lines[i + 1] if w["x1"] > x0 - 20 and w["x0"] < x1 + 200]
    return _JUNK.sub("", " ".join(w["text"] for w in ws)).strip() or None


def _clean(field, val):
    val = val.strip()
    if field.endswith("_id"):
        return val.split(" ")[0].upper()
    return re.split(r"\s{2,}|:", val)[0].strip()


def extract_fields(lines, page_no=None):
    """Fuzzy-find each FIELDS label on each line, take the value right of it (or below),
    validate with the regex. Returns {field: {value, score, page, how}} (best per field)."""
    all_labels = {_norm(l) for f in FIELDS.values() for l in f["labels"]}
    best = {}
    for i, line in enumerate(lines):
        text, spans = "", []
        for w in line:
            text += " " if text else ""
            spans.append((len(text), len(text) + len(w["text"])))
            text += w["text"]
        low = text.lower()
        for field, spec in FIELDS.items():
            if any(x in low for x in FIELD_EXCLUDE.get(field, [])):
                continue
            rx = re.compile(spec["regex"])
            for label in sorted(spec["labels"], key=len, reverse=True):  # 'applicant name' before 'name'
                al = fuzz.partial_ratio_alignment(label, low)
                if al is None or al.score < FUZZY_THRESHOLD:
                    continue
                if al.dest_start > 0 and low[al.dest_start - 1].isalnum():
                    continue  # label must start at a word boundary
                cands = []
                v = _right_value(line, spans, al.dest_end)
                if v:
                    cands.append((_clean(field, v), "right"))
                lw = [w for w, s in zip(line, spans) if s[0] < al.dest_end and s[1] > al.dest_start]
                if lw and not v:
                    vb = _below_value(lines, i, lw[0]["x0"], lw[-1]["x1"])
                    if vb:
                        cands.append((_clean(field, vb), "below"))
                for val, where in cands:
                    if val and rx.match(val) and _norm(val) not in all_labels \
                            and not (set(_norm(val).replace("-", " ").split()) & NOT_A_VALUE):
                        score = al.score / 100 * (1.0 if where == "right" else 0.9) * (len(label) / (len(label) + 4))
                        if field not in best or score > best[field]["score"]:
                            best[field] = {"value": val, "score": round(score, 3), "page": page_no, "how": where}
                        break
                if field in best and best[field]["page"] == page_no:
                    break
    return best

# =============================== PIPELINE ===================================
FIELD_NAMES = list(FIELDS)


def process_pdf(pdf_path, ocr_page, engine="rapid", max_pages=None, log=print):
    """Whole PDF -> result dict (pages, statements, transactions, fields, summary)."""
    t0 = time.time()
    n = page_count(pdf_path)
    n = min(n, max_pages) if max_pages else n
    pages, page_lines = [], {}
    for p in range(n):
        t = time.time()
        pg = get_page_words(pdf_path, p, ocr_page, engine)
        cls, lines = classify_page(pg["words"])
        page_lines[p] = lines
        ws = pg["words"]
        pages.append({"page": p + 1, "source": pg["source"], "skew_deg": round(pg["angle"], 2),
                      "n_words": len(ws), "mean_conf": round(sum(w["conf"] for w in ws) / len(ws), 3) if ws else None,
                      "seconds": round(time.time() - t, 2), **cls,
                      "text": "\n".join(line_text(l) for l in lines)})
        if log:
            log(f"  page {p+1}/{n}: {pg['source']:4s} {cls['label']:16s} st={cls['statement_score']:.2f} "
                f"form={cls['form_score']:.2f} words={len(ws)} {time.time()-t:.1f}s{' (cache)' if pg['cached'] else ''}")
    statements, txns = [], []
    for k, run in enumerate(group_runs([pg["label"] for pg in pages])):
        s, rows = analyse_statement(run, page_lines)
        s["statement"] = k + 1
        for r in rows:
            r["statement"] = k + 1
        statements.append(s)
        txns += rows
    # fields: non-statement pages first; names on statements are weaker evidence (x0.5)
    found = {}
    order = sorted(range(n), key=lambda p: pages[p]["label"] == "bank_statement")
    for p in order:
        for f, v in extract_fields(page_lines[p], page_no=p + 1).items():
            if pages[p]["label"] == "bank_statement":
                v["score"] = round(v["score"] * 0.5, 3)
            if f not in found or v["score"] > found[f]["score"]:
                found[f] = v
    res = {"file": os.path.basename(pdf_path), "engine": engine, "n_pages": n, "fields": found,
           "statements": statements, "transactions": txns, "pages": pages,
           "seconds": round(time.time() - t0, 1)}
    res["summary"] = summary_row(res)
    return res


def summary_row(r):
    st = r["statements"]
    checked = sum(s["rows_checked"] for s in st)
    ok = sum((s["balance_ok_rate"] or 0) * s["rows_checked"] for s in st)
    starts = [s["period_start"] or s["txn_start"] for s in st if s["txn_start"]]
    ends = [s["period_end"] or s["txn_end"] for s in st if s["txn_end"]]
    row = {"file": r["file"], "engine": r["engine"], "n_pages": r["n_pages"],
           "has_statements": bool(st),
           "statement_pages": " ".join(str(p) for s in st for p in s["pages"]),
           "n_statements": len(st),
           "months_covered": max([s["months"] for s in st], default=0),
           "distinct_txn_months": max([s["distinct_txn_months"] for s in st], default=0),
           "period_start": min(starts) if starts else None, "period_end": max(ends) if ends else None,
           "n_transactions": len(r["transactions"]),
           "balance_ok_rate": round(ok / checked, 3) if checked else None,
           "seconds": r["seconds"]}
    for f in FIELD_NAMES:
        row[f] = r["fields"].get(f, {}).get("value")
        row[f + "_score"] = r["fields"].get(f, {}).get("score")
    return row


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_folder(ocr_page, engine, data_dir="data", out_dir="outputs", max_pages=None):
    """Every PDF in data_dir -> outputs/<engine>/<pdf>.json, <pdf>_transactions.csv,
    summary.csv and a manual_review.csv template (never overwritten)."""
    pdfs = sorted(glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True))
    if not pdfs:
        print(f"No PDFs in {data_dir}/")
        return []
    out = os.path.join(out_dir, engine)
    summary = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf}")
        name = os.path.splitext(os.path.basename(pdf))[0]
        try:
            r = process_pdf(pdf, ocr_page, engine, max_pages)
        except Exception as e:  # one bad file (password, corrupt) must not stop the batch
            print(f"  ERROR {type(e).__name__}: {e}")
            summary.append({"file": os.path.basename(pdf), "engine": engine, "error": f"{type(e).__name__}: {e}"})
            continue
        save_json(r, os.path.join(out, name + ".json"))
        if r["transactions"]:
            save_csv(r["transactions"], os.path.join(out, name + "_transactions.csv"))
        summary.append(r["summary"])
        s = r["summary"]
        print(f"  -> statements={s['has_statements']} pages=[{s['statement_pages']}] months={s['months_covered']} "
              f"rows={s['n_transactions']} balance_ok={s['balance_ok_rate']} "
              f"fields={ {f: s[f] for f in FIELD_NAMES} } ({r['seconds']}s)")
    save_csv(summary, os.path.join(out, "summary.csv"))
    mr = os.path.join(out_dir, "manual_review.csv")
    if not os.path.exists(mr):
        save_csv([{"file": s["file"], **{c + "_true": "" for c in ["has_statements", "months_covered"] + FIELD_NAMES},
                   "notes": ""} for s in summary], mr)
        print(f"Wrote {mr}: fill the *_true columns by hand, then use evaluate()")
    print(f"Done -> {out}/summary.csv")
    return summary

# =============================== EVALUATE ===================================
def evaluate(summary_csv, manual_csv="outputs/manual_review.csv", name_fuzzy=90):
    """Accuracy per field vs your hand-filled manual_review.csv (only filled *_true cells count)."""
    def rd(p):
        with open(p, encoding="utf-8") as f:
            return {r["file"]: r for r in csv.DictReader(f)}
    pred, truth = rd(summary_csv), rd(manual_csv)
    n_ = lambda x: " ".join(str(x or "").lower().split())
    yes = ("1", "true", "yes", "y")
    out = []
    for field in ["has_statements", "months_covered"] + FIELD_NAMES:
        hits = []
        for file, t in truth.items():
            tv, pv = n_(t.get(field + "_true")), n_(pred.get(file, {}).get(field))
            if not tv:
                continue
            if field == "has_statements":
                ok = (tv in yes) == (pv in yes)
            elif field == "months_covered":
                ok = pv != "" and abs(float(tv) - float(pv)) < 0.5
            elif field.endswith("name"):
                ok = fuzz.ratio(tv, pv) >= name_fuzzy
            else:
                ok = tv.replace(" ", "") == pv.replace(" ", "")
            hits.append(ok)
        if hits:
            out.append({"field": field, "n": len(hits), "correct": sum(hits),
                        "accuracy": round(sum(hits) / len(hits), 3)})
    for r in out:
        print(r)
    return out
