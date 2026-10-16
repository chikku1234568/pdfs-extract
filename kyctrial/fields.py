"""Step 6: key-value extraction for application fields.

For every text line we fuzzy-search each label synonym (config['fields']).
When a label is found, the value is taken
   (a) to the RIGHT of the label on the same line (up to a big column gap), else
   (b) from the line directly BELOW, horizontally overlapping the label.
Candidates must pass the field's regex validator. Highest score wins.
Labels/regexes live in config.py so you can adapt them without code changes."""
import re
from rapidfuzz import fuzz

# a label for field X should not fire inside a line that belongs to field Y
EXCLUDE = {
    "name": ["co-app", "co app", "coapp", "co-borrower", "co borrower", "father", "spouse",
             "mother", "bank", "branch", "nominee", "account name"],
    "app_id": ["co-app", "co app", "coapp", "co-borrower"],
}
# words that mean we grabbed another label instead of a value
NOT_A_VALUE = {"date", "birth", "name", "applicant", "application", "pan", "mobile", "address",
               "father", "loan", "amount", "occupation", "id", "no", "number", "signature"}
JUNK = re.compile(r"^[\s:;\-–_.|]+|[\s:;\-–_.|]+$")


def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def _joined(line):
    """Join words and remember where each word starts in the joined string."""
    text, spans = "", []
    for w in line:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(w["text"])))
        text += w["text"]
    return text, spans


def _all_labels(cfg):
    return [_norm(l) for f in cfg["fields"].values() for l in f["labels"]]


def _right_value(line, spans, end_char, cfg):
    words = [(w, s) for w, s in zip(line, spans) if s[1] > end_char]
    if not words:
        return None, None
    h = sum(w["y1"] - w["y0"] for w, _ in words) / len(words)
    out, last_x1, first = [], None, None
    for w, (a, b) in words:
        t = w["text"]
        if a < end_char:  # word that contains the label end (e.g. "Name:Rahul")
            t = t[end_char - a:]
        if not JUNK.sub("", t):
            continue  # leftover of the label itself, e.g. ':' or 'No:'
        if out and w["x0"] - last_x1 > 3 * h:
            break  # big gap AFTER the value started = next column / next key
        # stop if a new label starts (e.g. "Name: X   Application No: Y")
        if out and any(fuzz.partial_ratio(lbl, _norm(t)) >= 95 and len(t) > 3 for lbl in ("application", "co-applicant", "mobile", "date", "pan")):
            break
        first = first or w
        out.append(t); last_x1 = w["x1"]
    val = JUNK.sub("", " ".join(out)).strip()
    return (val or None), first


def _below_value(lines, i, label_x0, label_x1):
    if i + 1 >= len(lines):
        return None
    ws = [w for w in lines[i + 1] if w["x1"] > label_x0 - 20 and w["x0"] < label_x1 + 200]
    return JUNK.sub("", " ".join(w["text"] for w in ws)).strip() or None


def _clean(field, val):
    if val is None:
        return None
    val = val.strip()
    if field in ("app_id", "co_app_id"):
        val = val.split(" ")[0].upper()
    else:
        val = re.split(r"\s{2,}|:", val)[0].strip()
    return val


def extract_fields(lines, cfg, page_no=None):
    """Returns {field: {value, score, page, bbox}} for the best candidate per field."""
    thr = cfg["fuzzy_threshold"]
    best = {}
    for i, line in enumerate(lines):
        text, spans = _joined(line)
        low = text.lower()
        for field, spec in cfg["fields"].items():
            if any(x in low for x in EXCLUDE.get(field, [])):
                continue
            rx = re.compile(spec["regex"])
            # longest label first so 'applicant name' beats 'name'
            for label in sorted(spec["labels"], key=len, reverse=True):
                al = fuzz.partial_ratio_alignment(label, low)
                if al is None or al.score < thr:
                    continue
                # label must start at a word boundary
                if al.dest_start > 0 and low[al.dest_start - 1].isalnum():
                    continue
                cands = []
                v, w = _right_value(line, spans, al.dest_end, cfg)
                if v:
                    cands.append((_clean(field, v), "right", w))
                lw = [w for w, s in zip(line, spans) if s[0] < al.dest_end and s[1] > al.dest_start]
                if lw and not v:  # only look below when nothing is to the right
                    vb = _below_value(lines, i, lw[0]["x0"], lw[-1]["x1"])
                    if vb:
                        cands.append((_clean(field, vb), "below", None))
                for val, where, w in cands:
                    if val and rx.match(val) and _norm(val) not in _all_labels(cfg) \
                            and not (set(_norm(val).replace("-", " ").split()) & NOT_A_VALUE):
                        score = al.score / 100 * (1.0 if where == "right" else 0.9) * (len(label) / (len(label) + 4))
                        if field not in best or score > best[field]["score"]:
                            box = None
                            if lw:
                                box = [lw[0]["x0"], lw[0]["y0"], max(x["x1"] for x in line), lw[0]["y1"]]
                            best[field] = {"value": val, "score": round(score, 3),
                                           "page": page_no, "how": where, "bbox": box}
                        break
                if field in best and best[field]["page"] == page_no:
                    break
    return best
