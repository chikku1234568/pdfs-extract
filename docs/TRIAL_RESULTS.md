# Results on synthetic fixtures (this dev box)

Environment: Linux x86_64, Python 3.13.5, CPU only (8 cores, also measured pinned to 4 cores), default settings (250 DPI).
These are **synthetic** files; real client scans will differ - treat this as a smoke test, not an accuracy claim.

| File | Expected | Got |
|---|---|---|
| synthetic_digital.pdf (text layer, 5 pages, 44 txns, Apr–Sep 2023) | statements p3–4, 6 months, 4 fields | **all correct**: pages 3–4, 6 months, 44/44 rows, reconcile 100%, all 4 fields exact |
| synthetic_scanned.pdf (150 dpi b/w, rotated, noisy; 45 txns, Jan–Jun 2024) | statements p3–4, 6 months | pages 3–4 ✓, 6 months ✓, 38/45 rows, reconcile 51.5%, app_id ✓, name ✓, co_app_id ✓, co_app_name **"YaminiBit"** vs "Yamini Bir" ✗ |
| synthetic_no_statement.pdf (scanned, 3 pages) | no statements | no statements ✓, name ✓, co_app_id ✓, app_id **missed**, co_app_name "Ira Sachdey" vs "Sachdev" ✗ |

What failed and why (scanned file):
- Statement period line OCR'd as "28i06/2024" → not parsed, months taken from transaction span (still 6).
- Slashes in dates read as `:`/`1`/missing → handled by a lenient first-column date parser; 7 rows still lost.
- Thousands separators read as `.`, `5`→`$`, digits dropped (".359.74" for "5,359.74") → ~half the rows fail the balance check. The check is doing its job: it flags exactly these rows.
- Name OCR errors (r→t, v→y) - fuzzy matching against ground truth will be needed when evaluating.

Speed (OCR pages, fresh run, no cache): ~1.0–2.6 s/page at 250 DPI on 8 cores; ~2.0–2.9 s/page at 300 DPI pinned to 4 cores.
Digital (text-layer) pages: ~0.1 s/page. Cached reruns: ~0.1 s/page. A 40-page scanned file ≈ 1–2 min.


## Update 2026-10-16: OCR engine + preprocessing change (see bench/OCR_RESEARCH.md)
Default engine is now RapidOCR 3.x with PP-OCRv5 mobile models (now `utils/ocr_rapid.py`), and preprocessing is deskew only (no binarisation, no median filter). Same fixtures, rerun:
- synthetic_scanned.pdf: pages 3–4 ✓, 6 months ✓, 44/45 rows, reconcile **76.3%** (was 51.5%), all 4 fields exact (co-applicant name now right)
- synthetic_no_statement.pdf: no statements ✓, all 4 fields exact (app_id was missed before)
- synthetic_digital.pdf: unchanged, all correct


## Update 2026-10-18: simple layout (`utils/` + `main_*.py` + `main.ipynb`)
Same logic, flatter code. Rerun with `python main_rapid.py --data tests/fixtures` / `python main_paddle.py --data tests/fixtures` and `python tests/smoke_test.py [paddle]` (all checks pass for both):

| File | RapidOCR v5 mobile | PaddleOCR v5 mobile |
|---|---|---|
| synthetic_digital.pdf | pages 3–4, 6 months, 44/44 rows, balance ok 100%, 4/4 fields | same (text layer, no OCR) |
| synthetic_scanned.pdf | pages 3–4, 6 months, 44/45 rows, balance ok **76.3%**, 4/4 fields, ~7 s | pages 3–4, 6 months, 45/45 rows, balance ok **80.0%**, 4/4 fields, ~10 s |
| synthetic_no_statement.pdf | no statements, 4/4 fields | no statements, 4/4 fields |

Note: the balance check column is now called `balance_ok` (was `reconciles`), and the per-file rate `balance_ok_rate` (was `reconcile_rate`).
