"""All tunable knobs in one place. Edit here (or pass --config my.yaml) instead
of hunting through modules. Keys in the YAML override these defaults."""
import copy
import yaml

DEFAULTS = {
    "dpi": 250,                    # render DPI for scanned pages (200-300 is the sweet spot)
    "min_text_chars": 80,          # a text layer with fewer chars is treated as "no text layer"
    "preprocess": {
        "deskew": True,
        "denoise": False,          # median filter; bench showed it hurts neural OCR on clean-ish scans
        "threshold": False,        # adaptive threshold -> pure B/W. OFF: binarising cut amount recall 0.99->0.74 with RapidOCR (bench/)
        "block_size": 31,          # adaptive threshold window (odd). Bigger = smoother
        "c": 15,                   # adaptive threshold constant. Bigger = lighter output
        "remove_borders": False,
        "max_skew_deg": 10,
    },
    "ocr": {
        "engine": "rapidocr_v5",   # rapidocr_v5 | rapidocr_v6 | rapidocr_v4 | tesseract | onnxtr (see bench/OCR_RESEARCH.md)
        "threads": 4,
        "min_conf": 0.5,
    },
    "classify": {
        "statement_min_score": 0.45,
        "form_min_score": 0.35,
    },
    "statements": {
        "balance_tolerance": 1.0,  # rupees
        "max_page_gap": 1,         # statement pages separated by <= this many other pages stay in one run
    },
    # Label synonyms for key-value extraction. Add the client's real labels here.
    "fields": {
        "app_id": {
            "labels": ["application id", "application no", "application number", "app id",
                       "app no", "loan application no", "lan", "reference no", "ref no"],
            "regex": r"^(?=.*\d)[A-Z0-9][A-Z0-9/\-]{5,24}$",
        },
        "name": {
            "labels": ["applicant name", "name of applicant", "name of the applicant",
                       "customer name", "borrower name", "name"],
            "regex": r"^[A-Za-z][A-Za-z .']{2,60}$",
        },
        "co_app_name": {
            "labels": ["co-applicant name", "co applicant name", "name of co-applicant",
                       "co-borrower name", "co borrower name"],
            "regex": r"^[A-Za-z][A-Za-z .']{2,60}$",
        },
        "co_app_id": {
            "labels": ["co-applicant id", "co applicant id", "co-applicant application id",
                       "co-applicant application no", "co-borrower id"],
            "regex": r"^(?=.*\d)[A-Z0-9][A-Z0-9/\-]{5,24}$",
        },
    },
    "fuzzy_threshold": 85,
}


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
