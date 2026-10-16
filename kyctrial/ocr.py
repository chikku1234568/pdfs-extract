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
_ENGINE_KEY = None

# Engines selectable with config `ocr.engine` (see bench/OCR_RESEARCH.md for the benchmark):
#   rapidocr_v5   - rapidocr>=3 with PP-OCRv5 mobile det + en rec (ONNX, CPU)  [recommended / default]
#   rapidocr_v6   - rapidocr>=3 with PP-OCRv6 det/rec (`ocr.model_type`: tiny|small)
#   rapidocr_v4   - rapidocr_onnxruntime 1.x, PP-OCRv4 bundled in the wheel (original kit engine)
#   paddleocr_v5  - PaddleOCR 3.x native PP-OCRv5 mobile (CPU); optional, see requirements-paddle.txt
#   tesseract     - Tesseract 5 via pytesseract (needs tesseract.exe installed)
#   onnxtr        - docTR models on ONNX Runtime (pip install "onnxtr[cpu]")
# rapidocr>=3 / paddleocr / onnxtr download model files on first use: run once with internet
# (or copy the model cache) before moving to the offline laptop.


def _threads(cfg):
    return int(cfg.get("threads", 4))


def _make_engine(cfg):
    name = cfg.get("engine", "rapidocr_v5")
    th = _threads(cfg)
    if name == "rapidocr_v4":
        from rapidocr_onnxruntime import RapidOCR
        eng = RapidOCR(intra_op_num_threads=th, inter_op_num_threads=1)
        return lambda img: [(b, t, float(c)) for b, t, c in (eng(img)[0] or [])]
    if name in ("rapidocr_v5", "rapidocr_v6"):
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import OCRVersion, ModelType, LangDet, LangRec
        v5 = name == "rapidocr_v5"
        mt = ModelType("mobile" if v5 else cfg.get("model_type", "small"))
        eng = RapidOCR(params={
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": th,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.ocr_version": OCRVersion.PPOCRV5 if v5 else OCRVersion.PPOCRV6, "Det.model_type": mt,
            "Det.lang_type": LangDet.CH,
            "Rec.ocr_version": OCRVersion.PPOCRV5 if v5 else OCRVersion.PPOCRV6, "Rec.model_type": mt,
            "Rec.lang_type": LangRec.EN if v5 else LangRec.CH})
        def run(img):
            r = eng(img)
            if r.boxes is None:
                return []
            return [(b, t, float(c)) for b, t, c in zip(r.boxes, r.txts, r.scores)]
        return run
    if name == "tesseract":
        import pytesseract
        os.environ.setdefault("OMP_THREAD_LIMIT", str(th))
        if cfg.get("tesseract_cmd"):
            pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_cmd"]
        def run(img):
            d = pytesseract.image_to_data(img, lang="eng", config=f"--oem 1 --psm {cfg.get('psm', 4)}",
                                          output_type=pytesseract.Output.DICT)
            out = []
            for t, x, y, w, h, c in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"]):
                if t.strip() and float(c) >= 0:
                    out.append(([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], t, float(c) / 100))
            return out
        return run
    if name == "paddleocr_v5":
        # Lazy import: paddlepaddle / paddleocr are NOT in requirements.txt.
        # Install with: pip install -r requirements-paddle.txt
        try:
            os.environ.setdefault("FLAGS_use_mkldnn", "1")
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "ocr.engine=paddleocr_v5 needs paddleocr + paddlepaddle. "
                "Install with: pip install -r requirements-paddle.txt"
            ) from e
        det = cfg.get("paddle_det", "PP-OCRv5_mobile_det")
        rec = cfg.get("paddle_rec", "en_PP-OCRv5_mobile_rec")
        eng = PaddleOCR(
            text_detection_model_name=det,
            text_recognition_model_name=rec,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            cpu_threads=th,
        )
        def run(img):
            import numpy as np
            bgr = np.stack([img] * 3, -1) if getattr(img, "ndim", 3) == 2 else img
            out = []
            for r in eng.predict(bgr):
                for poly, t, c in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
                    out.append((poly, t, float(c)))
            return out
        return run
    if name == "onnxtr":
        import numpy as np
        import onnxruntime as ort
        from onnxtr.models import ocr_predictor, EngineConfig
        so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
        ec = EngineConfig(providers=["CPUExecutionProvider"], session_options=so)
        pred = ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch="crnn_vgg16_bn",
                             det_engine_cfg=ec, reco_engine_cfg=ec, clf_engine_cfg=ec)
        def run(img):
            rgb = np.stack([img] * 3, -1) if img.ndim == 2 else img
            H, W = rgb.shape[:2]; out = []
            for b in pred([rgb]).pages[0].blocks:
                for l in b.lines:
                    for w in l.words:
                        (x0, y0), (x1, y1) = w.geometry
                        out.append(([(x0 * W, y0 * H), (x1 * W, y0 * H), (x1 * W, y1 * H), (x0 * W, y1 * H)],
                                    w.value, float(w.confidence)))
            return out
        return run
    raise ValueError(f"unknown ocr.engine: {name}")


def _engine(cfg=None):
    global _ENGINE, _ENGINE_KEY
    cfg = cfg or {}
    key = json.dumps(cfg, sort_keys=True, default=str)
    if _ENGINE is None or key != _ENGINE_KEY:
        _ENGINE, _ENGINE_KEY = _make_engine(cfg), key
    return _ENGINE


def run_ocr(img, min_conf=0.5, ocr_cfg=None):
    import numpy as np
    if getattr(img, "ndim", 3) == 2 and (ocr_cfg or {}).get("engine", "rapidocr_v5") != "tesseract":
        img = np.stack([img] * 3, -1)
    words = []
    for box, text, conf in _engine(ocr_cfg)(img):
        conf = float(conf)
        if conf < min_conf or not text.strip():
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        words += split_segment(text.strip(), float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)), round(conf, 3))
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
    raw = f"{os.path.abspath(pdf_path)}|{st.st_size}|{st.st_mtime}|{page_no}|{cfg['dpi']}|{json.dumps(cfg['preprocess'], sort_keys=True)}|{cfg['ocr'].get('engine', 'rapidocr_v5')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def ocr_cached(img, pdf_path, page_no, cfg, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_key(pdf_path, page_no, cfg) + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), True
    words = run_ocr(img, cfg["ocr"]["min_conf"], cfg["ocr"])
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


if __name__ == "__main__":  # python -m kyctrial.ocr [engine]  -> pre-download / warm up models while online
    import sys
    import numpy as np
    eng = sys.argv[1] if len(sys.argv) > 1 else "rapidocr_v5"
    _engine({"engine": eng})(np.full((64, 256, 3), 255, np.uint8))
    print("ready:", eng)
