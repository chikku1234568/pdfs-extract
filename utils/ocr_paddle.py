"""PaddleOCR 3.x with PP-OCRv5 mobile models (native Paddle, CPU). Use its own venv
(requirements-paddle.txt).

    from utils.ocr_paddle import ocr_page
    words = ocr_page(img)   # img: numpy gray or RGB/BGR -> [{text,x0,y0,x1,y1,conf}, ...]

First run downloads the models into %USERPROFILE%\\.paddlex\\official_models\\
(override with env var PADDLE_PDX_CACHE_HOME). Do it once while online:
    python utils/ocr_paddle.py"""
import os
import re
import numpy as np

THREADS = 4       # try 2 if RAM is tight
MIN_CONF = 0.5
DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "en_PP-OCRv5_mobile_rec"
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        os.environ.setdefault("FLAGS_use_mkldnn", "1")
        from paddleocr import PaddleOCR
        _ENGINE = PaddleOCR(text_detection_model_name=DET_MODEL, text_recognition_model_name=REC_MODEL,
                            use_doc_orientation_classify=False, use_doc_unwarping=False,
                            use_textline_orientation=False, device="cpu", cpu_threads=THREADS)
    return _ENGINE


def split_segment(text, x0, y0, x1, y1, conf):
    """Split a phrase box into words with proportional x-ranges (same as ocr_rapid)."""
    cw = (x1 - x0) / max(len(text), 1)
    return [{"text": m.group(0), "x0": x0 + m.start() * cw, "y0": y0, "x1": x0 + m.end() * cw,
             "y1": y1, "conf": conf} for m in re.finditer(r"\S+", text)]


def ocr_page(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    words = []
    for r in get_engine().predict(img):
        for poly, text, conf in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
            if float(conf) < MIN_CONF or not text.strip():
                continue
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            words += split_segment(text.strip(), float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)),
                                   round(float(conf), 3))
    return words


if __name__ == "__main__":  # warm up / download models while online
    ocr_page(np.full((64, 256, 3), 255, np.uint8))
    root = os.environ.get("PADDLE_PDX_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".paddlex"))
    print("ready: PaddleOCR PP-OCRv5 mobile. Models in", os.path.join(root, "official_models"))
