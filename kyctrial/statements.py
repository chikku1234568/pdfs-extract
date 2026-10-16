"""Step 5: turn bank-statement pages into transactions + month coverage.

Approach (no ML):
  1. group consecutive statement pages into runs (one run ~ one statement)
  2. find the header row (credit/debit/balance...) and use the x-positions of the
     header cells as column boundaries
  3. assign each word below the header to a column; a line that starts with a
     date starts a new transaction, other lines are description continuations
  4. parse amounts, then check prev_balance + credit - debit == balance
  5. months covered = distinct months of transaction dates (+ span earliest..latest);
     the 'period ... to ...' line is used for the period when present
"""
import re
from .classify import header_line
from .ocr import line_text
from .parsing import PERIOD_RE, DATE_RE, parse_date, parse_date_lenient, parse_amount, is_amount

COLUMN_SYNONYMS = {
    "date": ["date", "txn date", "tran date", "transaction date", "value date", "post date"],
    "description": ["description", "particulars", "narration", "details", "remarks", "transaction details"],
    "ref": ["chq", "cheque", "ref", "chq/ref", "reference", "instrument"],
    "debit": ["debit", "withdrawal", "withdrawals", "dr", "withdrawal amt", "debit amount"],
    "credit": ["credit", "deposit", "deposits", "cr", "deposit amt", "credit amount"],
    "balance": ["balance", "closing balance", "bal"],
}
NUMERIC = ("debit", "credit", "balance")
STOP_WORDS = ("closing balance", "total", "grand total", "end of statement", "page total")


def group_runs(page_labels, max_gap=1):
    """[ 'other','bank_statement','bank_statement','other','bank_statement' ] -> [[1,2,4]] (gap 1)."""
    runs, cur, last = [], [], None
    for i, lab in enumerate(page_labels):
        if lab != "bank_statement":
            continue
        if cur and i - last - 1 > max_gap:
            runs.append(cur); cur = []
        cur.append(i); last = i
    if cur:
        runs.append(cur)
    return runs


def _col_type(text):
    t = text.lower().strip(" .:()")
    for col in ("balance", "debit", "credit", "description", "ref", "date"):
        for syn in COLUMN_SYNONYMS[col]:
            if t == syn or t.startswith(syn) or (len(syn) > 3 and syn in t):
                return col
    return None


def header_columns(hline):
    """Merge header words into cells (small horizontal gaps) and type them."""
    hs = [w["y1"] - w["y0"] for w in hline]
    gap_tol = (sum(hs) / len(hs)) * 0.9
    cells = []
    for w in hline:
        if cells and w["x0"] - cells[-1]["x1"] < gap_tol and _col_type(w["text"]) is None:
            c = cells[-1]
            c["text"] += " " + w["text"]; c["x1"] = w["x1"]
        else:
            cells.append(dict(text=w["text"], x0=w["x0"], x1=w["x1"], y0=w["y0"], y1=w["y1"]))
    cols = []
    for c in cells:
        t = _col_type(c["text"])
        if t and t not in [x["type"] for x in cols]:
            cols.append({**c, "type": t})
    return cols


def assign_column(w, cols):
    """Numbers -> numeric column with the nearest right edge (amounts are right-aligned).
    Text  -> column whose span overlaps most / nearest centre."""
    if is_amount(w["text"]):
        num = [c for c in cols if c["type"] in NUMERIC]
        if num:
            return min(num, key=lambda c: abs(c["x1"] - w["x1"]))["type"]
    cx = (w["x0"] + w["x1"]) / 2
    return min(cols, key=lambda c: abs((c["x0"] + c["x1"]) / 2 - cx))["type"]


def parse_page_table(lines, cols=None):
    """Returns (rows, cols, header_bbox). cols from a previous page are reused
    when this page has no header (continuation pages)."""
    hi, hline = header_line(lines)
    hbox = None
    start = 0
    if hline:
        new_cols = header_columns(hline)
        if sum(c["type"] in NUMERIC for c in new_cols) >= 2:
            cols = new_cols
            start = hi + 1
            hbox = [min(w["x0"] for w in hline), min(w["y0"] for w in hline),
                    max(w["x1"] for w in hline), max(w["y1"] for w in hline)]
    if not cols:
        return [], None, None, None
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
        d = parse_date(" ".join(cells.get("date", []))) if "date" in cells else None
        if d is None and ln and ln[0]["x0"] < cols[0]["x1"] + 50 and not is_amount(ln[0]["text"]):
            d = parse_date_lenient(ln[0]["text"])  # OCR-damaged date in first column
        if "opening balance" in txt:
            amts = [parse_amount(w["text"]) for w in ln if is_amount(w["text"])]
            if amts:
                v, s = amts[-1]; opening = -v if s == "dr" else v
            continue
        if d is not None:
            row = {"date": d.isoformat(), "description": " ".join(cells.get("description", [])),
                   "debit": None, "credit": None, "balance": None,
                   "y0": min(w["y0"] for w in ln), "y1": max(w["y1"] for w in ln),
                   "raw": line_text(ln)}
            for col in NUMERIC:
                vals = [parse_amount(t) for t in cells.get(col, [])]
                vals = [v for v in vals if v[0] is not None]
                if vals:
                    v, s = vals[-1]
                    row[col] = -v if (col == "balance" and s == "dr") else v
            rows.append(row)
        elif rows and cells.get("description") and not any(is_amount(w["text"]) for w in ln):
            rows[-1]["description"] += " " + " ".join(cells["description"])  # wrapped narration
            rows[-1]["y1"] = max(w["y1"] for w in ln)
    if rows:
        x0 = min(c["x0"] for c in cols); x1 = max(c["x1"] for c in cols)
        tbox = [x0, (hbox or [0, rows[0]["y0"]])[1], x1, rows[-1]["y1"]]
    else:
        tbox = None
    return rows, cols, hbox, {"bbox": tbox, "opening": opening}


def reconcile(rows, opening=None, tol=1.0):
    """Mark each row ok/fail/na with the running-balance rule. First row checks
    against the opening balance if known, else is 'na'."""
    prev, ok, checked = opening, 0, 0
    for r in rows:
        r["reconciles"] = "na"
        if r["balance"] is None:
            continue
        if prev is not None:
            exp = prev + (r["credit"] or 0) - (r["debit"] or 0)
            checked += 1
            if abs(exp - r["balance"]) <= tol:
                ok += 1; r["reconciles"] = "ok"
            else:
                r["reconciles"] = "fail"
        prev = r["balance"]
    return (ok / checked) if checked else None, checked


def months_covered(dates):
    if not dates:
        return {"distinct_months": 0, "span_months": 0, "start": None, "end": None}
    ym = sorted({(d.year, d.month) for d in dates})
    lo, hi = min(dates), max(dates)
    span = (hi.year - lo.year) * 12 + hi.month - lo.month + 1
    return {"distinct_months": len(ym), "span_months": span,
            "start": lo.isoformat(), "end": hi.isoformat()}


def find_period(text):
    m = PERIOD_RE.search(text.replace("\n", " "))
    if not m:
        return None
    a, b = parse_date(m.group("a")), parse_date(m.group("b"))
    if a and b and a <= b:
        return a, b
    return None


def analyse_run(run_pages, page_lines, cfg):
    """run_pages: page indices; page_lines: {page: lines}. Returns run summary + rows."""
    cols, all_rows, boxes, opening = None, [], {}, None
    period = None
    for p in run_pages:
        lines = page_lines[p]
        period = period or find_period("\n".join(line_text(l) for l in lines[:25]))
        rows, cols, hbox, extra = parse_page_table(lines, cols)
        if extra and opening is None and extra["opening"] is not None and not all_rows:
            opening = extra["opening"]
        for r in rows:
            r["page"] = p
        all_rows += rows
        boxes[p] = {"header": hbox, "table": extra["bbox"] if extra else None}
    rate, checked = reconcile(all_rows, opening, cfg["statements"]["balance_tolerance"])
    from datetime import date
    m = months_covered([date.fromisoformat(r["date"]) for r in all_rows])
    if period:
        a, b = period
        m["period_start"], m["period_end"] = a.isoformat(), b.isoformat()
        m["period_months"] = (b.year - a.year) * 12 + b.month - a.month + 1
    return {"pages": run_pages, "n_rows": len(all_rows), "reconcile_rate": rate,
            "rows_checked": checked, **m}, all_rows, boxes
