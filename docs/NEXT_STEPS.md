# Next steps (if the baseline is not good enough)

Stay on-prem, CPU, open source. Check licences before client deployment.

## 1. OCR is weak (low `mean_conf`, low `balance_ok_rate`)
| Option | Notes | Licence |
|---|---|---|
| Tune first | `--dpi 300` (or `common.DPI = 300` in the notebook), look at the deskewed page in `main.ipynb` step 4, check words with `dump_page.py` | – |
| RapidOCR PP-OCRv6 tiny/small | change `Det/Rec.ocr_version` + `model_type` in `utils/ocr_rapid.py` `get_engine()`; faster, slightly less accurate (bench/) | Apache-2.0 |
| PaddleOCR server models | `DET_MODEL`/`REC_MODEL` in `utils/ocr_paddle.py` (e.g. `PP-OCRv5_server_det`); slower, sometimes better on bad scans | Apache-2.0 |
| docTR | strong on documents, PyTorch CPU, slower | Apache-2.0 |
| Tesseract 5 (+ pytesseract) | good on clean printed text, word-level boxes, separate Windows installer | Apache-2.0 |
Run all on the same 20 hand-checked pages and compare word accuracy + `balance_ok_rate`. A new engine only needs an `ocr_page(img)` function returning the same word dicts (copy `utils/ocr_rapid.py`).

## 2. Table structure is the bottleneck
- **PP-Structure** (PaddleOCR table recognition) – Apache-2.0, CPU ok.
- **Table Transformer (TATR)** detection + structure (Microsoft) – MIT, PyTorch CPU (a few s/page).
- Line-based: OpenCV morphology to find ruled table lines when statements have grid lines.
- Per-bank templates: once formats are known, fixed column x-ranges per bank are the most robust.

## 3. Page classification
- Train a TF-IDF + LogisticRegression (scikit-learn) on 200+ labelled pages: the page `text` and `label` are in each `outputs/<engine>/<pdf>.json` under `pages`. Fix the labels by hand, then fit in a notebook cell.
- Add image features (layout density, line counts) + gradient boosting.
- Later: **LayoutLMv3** / LiLT style layout-aware classifiers. Check licences: LayoutLMv3 weights are
  CC BY-NC-SA (non-commercial) – not usable commercially; LiLT / LayoutLM v1 are MIT; Donut is MIT.
  These are small encoder models, not generative LLMs – confirm with the client that this is acceptable.

## 4. Fields
- Collect real label variants → `FIELDS` at the top of `utils/common.py`.
- Checkbox / handwritten fields need different tooling (handwriting OCR such as TrOCR – MIT); scope with client first.

## 5. Engineering
- Parallel processing across PDFs (`multiprocessing`), keep 1 OCR process per 2 cores on 16 GB.
- Rotation detection (90/180°) via OCR confidence on rotated copies or Tesseract OSD.
- Encrypted PDFs: pypdfium2 accepts a `password=` argument.
- Licence notes: avoid PyMuPDF (AGPL) and Ghostscript (AGPL) unless a commercial licence is bought.
