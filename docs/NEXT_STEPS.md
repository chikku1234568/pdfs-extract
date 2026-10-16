# Next steps (if the baseline is not good enough)

Stay on-prem, CPU, open source. Check licences before client deployment.

## 1. OCR is weak (low `mean_conf`, low `reconcile_rate`)
| Option | Notes | Licence |
|---|---|---|
| Tune first | `--dpi 300`, threshold `c`/`block_size`, try `threshold: false` (grayscale into OCR), `--debug` images | – |
| RapidOCR newer det/rec models or the `rapidocr` package (v2+/v3) | PP-OCRv4/v5 ONNX models; API differs from `rapidocr-onnxruntime` | Apache-2.0 |
| PaddleOCR (full) | same family, more model choices, heavier install (paddlepaddle CPU) | Apache-2.0 |
| docTR | strong on documents, PyTorch CPU, slower | Apache-2.0 |
| Tesseract 5 (+ pytesseract) | good on clean printed text, word-level boxes, separate Windows installer | Apache-2.0 |
Run all on the same 20 hand-checked pages and compare word accuracy + `reconcile_rate`.

## 2. Table structure is the bottleneck
- **PP-Structure** (PaddleOCR table recognition) – Apache-2.0, CPU ok.
- **Table Transformer (TATR)** detection + structure (Microsoft) – MIT, PyTorch CPU (a few s/page).
- Line-based: OpenCV morphology to find ruled table lines when statements have grid lines.
- Per-bank templates: once formats are known, fixed column x-ranges per bank are the most robust.

## 3. Page classification
- Train the included TF-IDF + LogisticRegression on 200+ labelled pages (`kyctrial train`).
- Add image features (layout density, line counts) + gradient boosting.
- Later: **LayoutLMv3** / LiLT style layout-aware classifiers. Check licences: LayoutLMv3 weights are
  CC BY-NC-SA (non-commercial) – not usable commercially; LiLT / LayoutLM v1 are MIT; Donut is MIT.
  These are small encoder models, not generative LLMs – confirm with the client that this is acceptable.

## 4. Fields
- Collect real label variants → `config.py`.
- Checkbox / handwritten fields need different tooling (handwriting OCR such as TrOCR – MIT); scope with client first.

## 5. Engineering
- Parallel processing across PDFs (`multiprocessing`), keep 1 OCR process per 2 cores on 16 GB.
- Rotation detection (90/180°) via OCR confidence on rotated copies or Tesseract OSD.
- Encrypted PDFs: pypdfium2 accepts a `password=` argument.
- Licence notes: avoid PyMuPDF (AGPL) and Ghostscript (AGPL) unless a commercial licence is bought.
