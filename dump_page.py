"""What did OCR (or the text layer) read on ONE page, and what did the classifier make of it?

    python dump_page.py data\file.pdf 5                  # page 5 (1-based), RapidOCR
    python dump_page.py data\file.pdf 5 --engine paddle  # in the paddle venv

Writes outputs/dump_<pdfname>_p<N>.json with: engine, source, words (text, bbox, conf),
lines (grouped, top to bottom), and the classifier label / scores / features."""
import argparse
import os

from utils import common

ap = argparse.ArgumentParser()
ap.add_argument("pdf")
ap.add_argument("page", type=int, help="page number, 1-based")
ap.add_argument("--engine", choices=["rapid", "paddle"], default="rapid")
ap.add_argument("--out", default="outputs")
a = ap.parse_args()

if a.engine == "rapid":
    from utils.ocr_rapid import ocr_page
else:
    from utils.ocr_paddle import ocr_page

pg = common.get_page_words(a.pdf, a.page - 1, ocr_page, a.engine)
cls, lines = common.classify_page(pg["words"])
dump = {
    "pdf": os.path.basename(a.pdf), "page": a.page, "engine": a.engine,
    "source": pg["source"], "skew_deg": round(pg["angle"], 2), "from_cache": pg["cached"],
    "classification": cls,
    "lines": [{"y": round(min(w["y0"] for w in ln)), "text": common.line_text(ln),
               "words": [f"{w['text']} ({w['conf']:.2f})" for w in ln]} for ln in lines],
    "words": [{"text": w["text"], "bbox": [round(w["x0"]), round(w["y0"]), round(w["x1"]), round(w["y1"])],
               "conf": w["conf"]} for w in pg["words"]],
}
name = os.path.splitext(os.path.basename(a.pdf))[0]
path = os.path.join(a.out, f"dump_{name}_p{a.page}.json")
common.save_json(dump, path)
print(f"{cls['label']} st={cls['statement_score']} form={cls['form_score']} words={len(pg['words'])} -> saved {path}")
