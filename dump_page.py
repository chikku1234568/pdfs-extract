"""Show what OCR (or the text layer) read on one page, plus the classifier's view.
Usage: python dump_page.py data\file.pdf 5"""
import sys, cv2
from kyctrial import config, render, preprocess, ocr, classify
pdf, page = sys.argv[1], int(sys.argv[2]) - 1
cfg = config.load_config(None)
pg = render.load_page(pdf, page, cfg)
if pg.get("words"):
    words = pg["words"]
else:
    clean, angle = preprocess.preprocess(pg["image"], cfg, None, tag="dump")
    img = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR) if clean.ndim == 2 else clean
    words, _ = ocr.ocr_cached(img, pdf, page, cfg, "outputs/_ocr_cache")
print(f"source={pg['source']}  words={len(words)}\n")
lines = {}
for w in sorted(words, key=lambda w: (w["y0"], w["x0"])):
    key = next((k for k in lines if abs(k - w["y0"]) < 10), w["y0"])
    lines.setdefault(key, []).append(w)
for y in sorted(lines):
    print(f"{int(y):5d}  " + "  ".join(f"{w['text']}({w['conf']:.2f})" for w in sorted(lines[y], key=lambda w: w["x0"])))
res, _ = classify.classify_page(words, cfg)
print("\n", res["label"], "st=", res["statement_score"], "form=", res["form_score"])
print({k: v for k, v in res["features"].items()})
