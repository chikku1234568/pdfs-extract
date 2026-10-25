"""RapidOCR 3.x with PP-OCRv5 mobile models (ONNX Runtime, CPU, offline after first download).

    from utils.ocr_rapid import ocr_page
    words = ocr_page(img)   # img: numpy gray or RGB/BGR -> [{text,x0,y0,x1,y1,conf}, ...]

First run downloads 3 small .onnx files into <venv>/Lib/site-packages/rapidocr/models/.
Do it once while online:  python utils/ocr_rapid.py"""
import re
import numpy as np

THREADS = 4       # CPU threads for ONNX Runtime (benchmark used 4)
MIN_CONF = 0.5    # drop text boxes below this confidence
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import OCRVersion, ModelType, LangDet, LangRec
        _ENGINE = RapidOCR(params={
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": THREADS,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType("mobile"), "Det.lang_type": LangDet.CH,
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType("mobile"), "Rec.lang_type": LangRec.EN,
        })
    return _ENGINE


def split_segment(text, x0, y0, x1, y1, conf):
    """OCR often returns a whole phrase / table row as one box. Split it into words and
    give each an x-range proportional to its character position (good enough for columns)."""
    cw = (x1 - x0) / max(len(text), 1)
    return [{"text": m.group(0), "x0": x0 + m.start() * cw, "y0": y0, "x1": x0 + m.end() * cw,
             "y1": y1, "conf": conf} for m in re.finditer(r"\S+", text)]


def ocr_page(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    r = get_engine()(img)
    if r.boxes is None:
        return []
    words = []
    for box, text, conf in zip(r.boxes, r.txts, r.scores):
        if float(conf) < MIN_CONF or not text.strip():
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        words += split_segment(text.strip(), float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)),
                               round(float(conf), 3))
    return words


if __name__ == "__main__":  # warm up / download models while online
    import os, rapidocr
    ocr_page(np.full((64, 256, 3), 255, np.uint8))
    print("ready: RapidOCR PP-OCRv5 mobile. Models in", os.path.join(os.path.dirname(rapidocr.__file__), "models"))
