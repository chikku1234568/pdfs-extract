"""Uniform wrappers: each engine(img) -> list of kit-format words
{"text","x0","y0","x1","y1","conf"}. img = numpy uint8 gray or BGR. CPU only, 4 threads."""
import os, re
import numpy as np, cv2

THREADS = int(os.environ.get("OCR_THREADS", "4"))


def _split(text, x0, y0, x1, y1, conf):  # same as kit ocr.split_segment
    n = max(len(text), 1); cw = (x1 - x0) / n
    return [{"text": m.group(0), "x0": x0 + m.start() * cw, "y0": y0, "x1": x0 + m.end() * cw,
             "y1": y1, "conf": conf} for m in re.finditer(r"\S+", text)]


def _poly(box, text, conf):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    return _split(text.strip(), float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)), float(conf))


def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def rapid_legacy():  # rapidocr_onnxruntime 1.x = what the kit uses (PP-OCRv4 mobile, bundled)
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR(intra_op_num_threads=THREADS, inter_op_num_threads=1)
    def run(img):
        res, _ = eng(bgr(img)); out = []
        for box, t, c in res or []:
            if float(c) >= 0.5 and t.strip(): out += _poly(box, t, c)
        return out
    return run


def rapid3(det, rec, model_type_det, model_type_rec, lang_det="ch", lang_rec="ch"):
    def make():
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import OCRVersion, ModelType, LangDet, LangRec
        V = {"v4": OCRVersion.PPOCRV4, "v5": OCRVersion.PPOCRV5, "v6": OCRVersion.PPOCRV6}
        eng = RapidOCR(params={
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": THREADS,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.ocr_version": V[det], "Det.model_type": ModelType(model_type_det), "Det.lang_type": LangDet(lang_det),
            "Rec.ocr_version": V[rec], "Rec.model_type": ModelType(model_type_rec), "Rec.lang_type": LangRec(lang_rec)})
        def run(img):
            r = eng(bgr(img)); out = []
            if r.boxes is None: return out
            for box, t, c in zip(r.boxes, r.txts, r.scores):
                if c >= 0.5 and t.strip(): out += _poly(box, t, c)
            return out
        return run
    return make


def tesseract(psm=6):
    def make():
        import pytesseract
        os.environ["OMP_THREAD_LIMIT"] = str(THREADS)
        def run(img):
            d = pytesseract.image_to_data(img, lang="eng", config=f"--oem 1 --psm {psm}",
                                          output_type=pytesseract.Output.DICT)
            out = []
            for t, x, y, w, h, c in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"]):
                if t.strip() and float(c) >= 30:
                    out.append({"text": t, "x0": x, "y0": y, "x1": x + w, "y1": y + h, "conf": float(c) / 100})
            return out
        return run
    return make


def onnxtr(rec="crnn_vgg16_bn"):
    def make():
        from onnxtr.models import ocr_predictor, EngineConfig
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = THREADS; so.inter_op_num_threads = 1
        cfg = EngineConfig(providers=["CPUExecutionProvider"], session_options=so)
        pred = ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch=rec, det_engine_cfg=cfg,
                             reco_engine_cfg=cfg, clf_engine_cfg=cfg, assume_straight_pages=False,
                             detect_orientation=False, straighten_pages=False)
        def run(img):
            im = cv2.cvtColor(bgr(img), cv2.COLOR_BGR2RGB); H, W = im.shape[:2]
            doc = pred([im]); out = []
            for b in doc.pages[0].blocks:
                for l in b.lines:
                    for w in l.words:
                        g = np.array(w.geometry)
                        if g.ndim == 1: g = g.reshape(2, 2)
                        if w.confidence >= 0.3 and w.value.strip():
                            out.append({"text": w.value, "x0": float(g[:, 0].min() * W), "y0": float(g[:, 1].min() * H),
                                        "x1": float(g[:, 0].max() * W), "y1": float(g[:, 1].max() * H), "conf": float(w.confidence)})
            return out
        return run
    return make


def easyocr_():
    import torch; torch.set_num_threads(THREADS)
    import easyocr
    r = easyocr.Reader(["en"], gpu=False, verbose=False)
    def run(img):
        out = []
        for box, t, c in r.readtext(img, paragraph=False):
            if c >= 0.3 and t.strip(): out += _poly(box, t, c)
        return out
    return run


def paddle(version="PP-OCRv5", det="PP-OCRv5_mobile_det", rec="en_PP-OCRv5_mobile_rec"):
    def make():
        os.environ["FLAGS_use_mkldnn"] = "1"
        from paddleocr import PaddleOCR
        eng = PaddleOCR(text_detection_model_name=det, text_recognition_model_name=rec,
                        use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=False, device="cpu", cpu_threads=THREADS)
        def run(img):
            res = eng.predict(bgr(img)); out = []
            for r in res:
                for poly, t, c in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
                    if c >= 0.5 and t.strip(): out += _poly(poly, t, c)
            return out
        return run
    return make


ENGINES = {
    "rapidocr_v4_legacy": rapid_legacy,                       # kit default today
    "rapidocr_v5_mobile": rapid3("v5", "v5", "mobile", "mobile", "ch", "en"),
    "rapidocr_v5_server": rapid3("v5", "v5", "server", "server", "ch", "ch"),   # ~100 s/page on CPU: excluded
    "rapidocr_v5_srvrec": rapid3("v5", "v5", "mobile", "server", "ch", "ch"),  # mobile det + server rec
    "rapidocr_v6_small": rapid3("v6", "v6", "small", "small", "ch", "ch"),
    "rapidocr_v6_medium": rapid3("v6", "v6", "medium", "medium", "ch", "ch"),
    "rapidocr_v6_tiny": rapid3("v6", "v6", "tiny", "tiny", "ch", "ch"),
    "tesseract_psm6": tesseract(6),
    "tesseract_psm4": tesseract(4),
    "onnxtr_crnn": onnxtr("crnn_vgg16_bn"),
    "onnxtr_parseq": onnxtr("parseq"),
    "easyocr": easyocr_,
    "paddleocr_v5_mobile": paddle(),
    "paddleocr_v6_small": paddle(det="PP-OCRv6_small_det", rec="PP-OCRv6_small_rec"),
}
