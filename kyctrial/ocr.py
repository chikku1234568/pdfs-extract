"""Step 3: OCR with RapidOCR (PaddleOCR models via ONNX Runtime, CPU, offline).

Common word format used everywhere (same as the text-layer path):
    {"text": str, "x0": float, "y0": float, "x1": float, "y1": float, "conf": float}
Note: RapidOCR returns *text segments* (often a whole cell / phrase), not single
words. That is fine for tables because columns are separated by whitespace.
Results are cached as JSON per page so reruns skip OCR."""
import hashlib
import json
import os
import re

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR  # models ship inside the wheel
        _ENGINE = RapidOCR()
    return _ENGINE


def run_ocr(img, min_conf=0.5):
    result, _ = _engine()(img)
    words = []
    for box, text, conf in result or []:
        conf = float(conf)
        if conf < min_conf or not text.strip():
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        words += split_segment(text.strip(), min(xs), min(ys), max(xs), max(ys), round(conf, 3))
    return words


def split_segment(text, x0, y0, x1, y1, conf):
    """RapidOCR often returns a whole table row as ONE segment. Split it into
    whitespace tokens and give each an x-range proportional to its character
    position (monospace approximation - good enough for column assignment)."""
    n = max(len(text), 1)
    cw = (x1 - x0) / n
    out = []
    for m in re.finditer(r"\S+", text):
        out.append({"text": m.group(0), "x0": x0 + m.start() * cw, "y0": y0,
                    "x1": x0 + m.end() * cw, "y1": y1, "conf": conf})
    return out


def cache_key(pdf_path, page_no, cfg):
    st = os.stat(pdf_path)
    raw = f"{os.path.abspath(pdf_path)}|{st.st_size}|{st.st_mtime}|{page_no}|{cfg['dpi']}|{json.dumps(cfg['preprocess'], sort_keys=True)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def ocr_cached(img, pdf_path, page_no, cfg, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_key(pdf_path, page_no, cfg) + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), True
    words = run_ocr(img, cfg["ocr"]["min_conf"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(words, f)
    return words, False


def group_lines(words, y_tol=0.5):
    """Group words into text lines by vertical overlap. Returns list of lists,
    top-to-bottom, each sorted left-to-right."""
    lines = []
    for w in sorted(words, key=lambda w: (w["y0"] + w["y1"]) / 2):
        cy = (w["y0"] + w["y1"]) / 2
        h = max(w["y1"] - w["y0"], 1)
        for line in lines:
            if abs(line["cy"] - cy) < y_tol * max(h, line["h"]):
                line["words"].append(w)
                n = len(line["words"])
                line["cy"] = (line["cy"] * (n - 1) + cy) / n
                break
        else:
            lines.append({"cy": cy, "h": h, "words": [w]})
    lines.sort(key=lambda l: l["cy"])
    return [sorted(l["words"], key=lambda w: w["x0"]) for l in lines]


def line_text(line):
    return " ".join(w["text"] for w in line)
