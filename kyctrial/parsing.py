"""Shared regexes + parsers for dates and amounts (Indian formats)."""
import re
from datetime import date
from dateutil import parser as dparser

MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
# dd/mm/yyyy, dd-mm-yy, dd.mm.yyyy, dd-Mon-yyyy, dd Mon yyyy, dd-Mon-yy
DATE_RE = re.compile(
    rf"\b(\d{{1,2}}[/\-.]\d{{1,2}}[/\-.](?:\d{{4}}|\d{{2}})"
    rf"|\d{{1,2}}[\-\s/]?(?:{MONTHS})[a-z]*[\-\s/,]*(?:\d{{4}}|\d{{2}}))\b",
    re.I)
# 1,23,456.78 | 123456.78 | 1,234.00 Cr | 500.00Dr  (needs 2 decimals to avoid IDs/years)
AMOUNT_RE = re.compile(r"^-?\(?[₹]?\s?\d{1,3}(?:,\d{2,3})*(?:\.\d{2})\)?\s?(?:cr|dr)?\.?$|^-?\d+\.\d{2}\s?(?:cr|dr)?\.?$", re.I)
_D = DATE_RE.pattern.replace("\\b(", "\\b(?:", 1)  # same date pattern, non-capturing
PERIOD_RE = re.compile(
    rf"(?:period|from|statement\s+for)\b[^0-9]{{0,40}}(?P<a>{_D})\s*(?:to|till|-|–)\s*(?P<b>{_D})",
    re.I)


def parse_date(s):
    """Parse an Indian-style date (day first). Returns datetime.date or None."""
    s = s.strip().replace(",", " ")
    m = DATE_RE.search(s)
    if not m:
        return None
    try:
        d = dparser.parse(m.group(0).replace(".", "/"), dayfirst=True, fuzzy=True,
                          default=dparser.parse("2000-01-01")).date()
    except (ValueError, OverflowError):
        return None
    if not (1990 <= d.year <= date.today().year + 1):
        return None
    return d


def normalise_amount_text(s):
    """Fix common OCR slips in amounts: '117.886.29' -> '117,886.29',
    'C1'/'Ct'/'C7' -> 'Cr', stray ':' '$' -> removed. Extend as you see new errors."""
    t = s.strip().replace(" ", "").replace("₹", "").strip(":;'`|!+").replace("'", "")
    # letter->digit confusions, only when the token is mostly digits
    if sum(ch.isdigit() for ch in t) >= max(3, len(t) // 2):
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


_LENIENT_DATE = re.compile(r"^(\d{1,2})[^\d]?(\d{2})(?:[^\d]|1)?(\d{4}|\d{2})$")


def parse_date_lenient(token):
    """OCR-tolerant fallback for the date cell of a table row:
    '02:01:24', '260124', '24/01124' (slash read as 1) -> date. Use only on the
    first token of a table row, never on free text."""
    t = token.strip(" .:;,|")
    d = parse_date(t)
    if d:
        return d
    m = _LENIENT_DATE.match(t)
    if not m:
        return None
    dd, mm, yy = (int(x) for x in m.groups())
    if yy < 100:
        yy += 2000
    try:
        d = date(yy, mm, dd)
    except ValueError:
        return None
    return d if 1990 <= d.year <= date.today().year + 1 else None


def is_amount(s):
    return bool(AMOUNT_RE.match(normalise_amount_text(s)))


def parse_amount(s):
    """'1,23,456.78 Cr' -> (123456.78, 'cr'). Returns (None, None) if not an amount."""
    t = normalise_amount_text(s)
    if not AMOUNT_RE.match(t):
        return None, None
    sign = None
    low = t.lower().rstrip(".")
    if low.endswith("cr"):
        sign, low = "cr", low[:-2]
    elif low.endswith("dr"):
        sign, low = "dr", low[:-2]
    neg = low.startswith("-") or low.startswith("(")
    val = float(low.strip("-()").replace(",", ""))
    return (-val if neg else val), sign
