"""Run the whole folder with RapidOCR PP-OCRv5 mobile.

    python main_rapid.py                       # every PDF in data/
    python main_rapid.py --data tests/fixtures --max-pages 5

Writes outputs/rapid/<pdf>.json, <pdf>_transactions.csv, summary.csv (+ outputs/manual_review.csv)."""
import argparse

from utils import common
from utils.ocr_rapid import ocr_page

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data", help="folder with PDFs (default data/)")
ap.add_argument("--out", default="outputs")
ap.add_argument("--max-pages", type=int, default=None, help="only the first N pages per PDF (quick look)")
ap.add_argument("--dpi", type=int, default=common.DPI)
ap.add_argument("--no-cache", action="store_true", help="always re-run OCR")
a = ap.parse_args()

common.DPI = a.dpi
if a.no_cache:
    common.CACHE_DIR = None
common.run_folder(ocr_page, "rapid", a.data, a.out, a.max_pages)
