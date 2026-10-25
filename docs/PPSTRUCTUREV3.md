# PP-StructureV3: what it is and where it fits

This is a research note. PP-StructureV3 has **not** been installed or run on our pages yet. Numbers marked "official" come from the PaddleOCR docs. Anything marked *(guess)* is our own estimate.

## 1. What it is
- PP-StructureV3 is the **document-parsing pipeline in PaddleOCR 3.x** (Baidu, Apache-2.0). It was built with PaddleX and runs on the PaddlePaddle inference engine.
- It takes an image or PDF page and returns structured output: layout blocks in reading order, text, tables (HTML/XLSX), formulas, seals and charts, as Markdown, JSON or Word.
- It is a chain of small models, not one big model. OCR inside it is PP-OCRv5, the same family as our RapidOCR models.
- Compared with plain PaddleOCR (`PaddleOCR()`): that gives text lines and boxes only. PP-StructureV3 adds layout detection, table structure, reading order and export on top.
- Compared with the old PP-Structure / V2 (PaddleOCR 2.x, `PPStructure`): V3 has new layout models (PP-DocLayout family), SLANeXt and RT-DETR table-cell models, wired/wireless table routing, multi-column reading order, chart parsing and Markdown export. The API also changed to `from paddleocr import PPStructureV3`.

## 2. Capabilities (from the official docs)
- **Layout detection**: PP-DocLayout_plus-L (20 classes: text, title, table, header, footer, seal, image, chart…). Lighter options are PP-DocLayout-M/S and PicoDet table-only (`PicoDet_layout_1x_table`).
- **Region detection**: PP-DocBlockLayout, which groups blocks for multi-column reading order.
- **OCR**: PP-OCRv5 det/rec, server or mobile. English-only models such as `en_PP-OCRv4_mobile_rec` can be swapped in. The docs themselves say the default Chinese+English model has "limited accuracy for purely English texts".
- **Optional doc preprocessing**: page orientation (0/90/180/270) with PP-LCNet_x1_0_doc_ori, and unwarping with UVDoc (slow on CPU, about 870 ms officially).
- **Table recognition**:
  - A classifier first decides wired or wireless (PP-LCNet_x1_0_table_cls).
  - It then runs a structure model (SLANeXt_wired / wireless) and a cell detector (RT-DETR-L wired / wireless cell det).
  - OCR is re-run per cell (`use_ocr_results_with_table_cells`), and there is table orientation correction.
  - Output is `pred_html` per table, plus `save_to_xlsx()` / `save_to_html()`.
- **Formula recognition**: PP-FormulaNet / UniMERNet. Heavy and not needed for us.
- **Seal (stamp) text recognition**: optional, and could be useful for bank stamps.
- **Chart parsing**: PP-Chart2Table, a 0.58B multimodal model of about 1.4 GB. Not needed.
- **Outputs**: `print()`, `save_to_json`, `save_to_markdown`, `save_to_word`, `save_to_xlsx`, `save_to_html`, `save_to_img` (visualisations), and `concatenate_markdown_pages` for whole PDFs.
- **Interfaces**:
  - Python: `PPStructureV3(...).predict()` / `predict_iter()` (a generator, better for memory).
  - CLI: `paddleocr pp_structurev3 -i file.pdf`.
  - An optional local HTTP service (self-hosted; nothing goes to a cloud).
- **Engines**: Paddle (default), `onnxruntime` (some models are not supported yet, and the docs say to switch formula off) and `transformers`. On CPU it uses MKL-DNN by default (`enable_mkldnn=True`, `cpu_threads=10`).
- **Fine-tuning**: every module can be trained on our own data (layout, table and rec models).

## 3. Fit per pipeline component
| # | Component | Fit | Why |
|---|---|---|---|
| 1 | Ingest (pypdfium2, pdfplumber) | Not a fit | Keep our ingest. PP-StructureV3 can read PDFs, but we already render pages and use the text layer. Pass it rendered numpy images instead. |
| 2 | Preprocess + OCR | Partial | Its OCR is PP-OCRv5, the same as ours, so it adds no accuracy and costs Paddle overhead. Doc orientation classification is worth borrowing for rotated scans. |
| 3 | Page classification | Partial | Layout output (a `table` block covering most of the page, `seal`, form-like `text` blocks) can be an extra **feature** for the `st`/`form` score. It does not know "bank statement" versus "application form", so it is not a classifier by itself. |
| 4 | Lead ID extraction | Not a fit / small help | No key-value extraction. Application tables come back as HTML cells, which could make "label → value" lookup easier on digital-style forms. Regex + RapidFuzz still does the work. |
| 5 | Group statement pages | Not a fit | No cross-page logic. The header block from layout may make account-number lookup a bit cleaner. |
| 6 | Table extraction (Path A) | **Best fit** | Table detection, wired/wireless structure, per-cell OCR, and HTML/XLSX output, which `pandas.read_html` turns straight into a dataframe. Our balance check is still the quality gate. |
| 6 | Path B (VLM) | Alternative | It is a cheaper middle ground than a VLM: small models on CPU. The newer PaddleOCR-VL is the Paddle VLM option and should be judged separately. |
| 7 | Month coverage | Not a fit | Pure logic over dates. Keep ours. |
| 8 | Lead-level Excel | Not a fit | Keep pandas/openpyxl. `save_to_xlsx` is per-table debug output only. |

## 4. Where to try it
1. **Step 6, as Path A2.** Run it only on pages already classified as statements, with table recognition on and everything else off. Parse `table_res_list[*].pred_html` → pandas → map columns (Date / Narration / Debit / Credit / Balance) → `balance_ok`. Compare against the current rules on the same pages.
2. **Optional, step 3 feature.** Use the share of the page area covered by `table` blocks from the cheap layout model (PP-DocLayout-S/M or the PicoDet table model) as an extra signal for uncertain pages only.
3. Leave all other components as they are.

## 5. Pros and cons vs our paths
**Versus Path A (RapidOCR + rules / img2table)**
- Pros:
  - It learns column and row structure, so it should cope better with borderless (wireless) tables, wrapped narration lines and multi-line cells *(guess, needs testing)*.
  - Output arrives as HTML/XLSX.
  - It is Apache-2.0 and can be fine-tuned.
- Cons:
  - Much slower: several heavy models per page.
  - It needs a separate Paddle venv on Python 3.12.
  - It is trained mostly on Chinese/English papers, magazines and reports, not noisy B&W bank scans.
  - Cell merges and splits on faint ruling lines are a known weakness of table-structure models in general.
  - It does not fix OCR errors in amounts.
  - The default rec model is weaker for English, so it must be swapped.
- Takeaway: Rules with per-bank column x-ranges plus the balance check may still win on known formats.

**Versus Path B (local VLM)**
- Pros: far lighter than Qwen2.5-VL-3B on CPU; deterministic boxes; no hallucinated numbers (output is grounded in OCR); a confidence score per text line.
- Cons: no semantic understanding of headers or the Dr/Cr meaning. Column mapping is still our code.

## 6. Feasibility on the client laptop (Windows, 16 GB RAM, CPU only)
- **Runs?** Yes. CPU is supported, and Paddle CPU wheels exist for Windows with Python 3.9–3.13. Everything is offline once the models are downloaded or copied, and no cloud is needed.
- **Speed (official, model inference only, Xeon Gold 6271C at 8 threads, so excluding pre/post-processing):**
  - PP-DocLayout_plus-L: about 635 ms
  - PP-DocBlockLayout: about 506 ms
  - PP-OCRv5_server_det: about 383 ms
  - RT-DETR-L cell det: about 403 ms per table
  - SLANeXt_wired: about 502 ms per table
- **Full default pipeline:** expect several seconds to 10+ s per page on a laptop CPU *(guess)*, versus our 1.76 s/page with RapidOCR. With 40 PDFs × about 30 pages it is fine only if limited to statement pages.
- **RAM:** the full default pipeline loads many models. The docs warn about OOM and slowness and suggest disabling features and using lighter models. With formula, chart, seal and unwarping off, it should fit in 16 GB *(guess, measure peak RSS)*.
- **Install** (in the existing `.venv-paddle`, Python 3.12):
  ```bat
  py -3.12 -m venv .venv-paddle
  .venv-paddle\Scripts\activate
  python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  python -m pip install "paddleocr[doc-parser]"
  ```
  Models download to `~/.paddlex/official_models` on first run. On an air-gapped client laptop, download them once elsewhere and point the `*_model_dir` arguments at that folder.
- **Verdict:** go for a trial, limited to statement pages in step 6. It is not a replacement for the whole pipeline.

## 7. Trial plan (flat style: one extra file `main_ppstruct.py`, `.venv-paddle`)
```python
# main_ppstruct.py  - statement pages only; writes JSON + per-page tables
import json, io, pandas as pd, pypdfium2 as pdfium, numpy as np
from paddleocr import PPStructureV3

pipe = PPStructureV3(
    lang="en",                                   # English rec instead of the zh+en default
    layout_detection_model_name="PP-DocLayout-M", # lighter than plus-L; compare both
    use_doc_orientation_classify=True,
    use_doc_unwarping=False, use_textline_orientation=False,
    use_formula_recognition=False, use_chart_recognition=False,
    use_seal_recognition=False, use_table_recognition=True,
    device="cpu", cpu_threads=8,
)

def tables_for_page(pdf_path, page_no, scale=200/72):
    img = pdfium.PdfDocument(pdf_path)[page_no].render(scale=scale).to_numpy()  # RGB
    res = next(iter(pipe.predict(img[:, :, ::-1])))        # BGR like OpenCV
    dfs = [pd.read_html(io.StringIO(t["pred_html"]))[0] for t in res.json["res"].get("table_res_list", [])]
    return res.json["res"], dfs
```
Steps:
1. Pick 20–30 statement pages that already have hand-checked transactions (clean, faint, borderless and multi-line narration).
2. Run the script, then save the JSON in `outputs/ppstruct/` (git-ignored, like the other outputs).
3. Map the columns with the existing fuzzy header logic, then run `balance_ok`.
4. Compare against current Path A: rows found, amount accuracy, balance-check pass rate, seconds per page, and peak RAM.
5. Repeat with `PP-DocLayout_plus-L` and with the cell-to-HTML flags (`use_wired_table_cells_trans_to_html=True`).
6. Decide: keep it as the fallback for pages where rules fail `balance_ok`, or drop it.

Quick CLI smoke test: `paddleocr pp_structurev3 -i page.png --use_formula_recognition False --use_chart_recognition False --device cpu --save_path out`

## 8. Sources
- PP-StructureV3 usage (models, parameters, CPU timings, API): https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html
- PaddleOCR repo (Apache-2.0): https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR 3.x installation (`doc-parser` extra): https://www.paddleocr.ai/latest/en/version3.x/installation.html
- PaddlePaddle install (CPU, Windows): https://www.paddlepaddle.org.cn/en/install/quick
- PaddleOCR 3.0 technical report (arXiv 2507.05595): https://arxiv.org/abs/2507.05595
