# OCR engine research and benchmark (offline, CPU, Windows target)

Date: 2026-10-16. Target: Windows laptop, 16 GB RAM, no GPU, fully offline, open source, no LLMs and no external APIs.
Documents: English, poor black-and-white scans of Indian bank statements (date/description/debit/credit/balance tables) and loan/KYC application forms.

**Bottom line**
- **Use RapidOCR 3.x with PP-OCRv5 mobile models (`ocr.engine: rapidocr_v5`) and deskew-only preprocessing.**
- It tied for best accuracy: 98.6–99.5% amount recall and 99.3% running-balance reconcile on the A/B statement layouts.
- It also used the least RAM of the accurate engines (~0.7 GB) and took ~1.8 s per page on 4 CPU threads.
- PaddleOCR 3.x (same models, native runtime) matched its accuracy but was 2.4× slower and used 3× the RAM.
- docTR/OnnxTR came a close third.
- Tesseract and EasyOCR were clearly worse on these scans.
- **The biggest single gain was not the engine: it was turning off binarisation.** The kit's adaptive threshold cut RapidOCR amount recall from 0.99 to 0.74. The kit default is now deskew only. A skew-detection bug on speckled scans is also fixed.

## 1. Survey

Licence covers the code / the model weights. "LLM" = vision-language-model OCR (out of scope).

| Engine | What it is | Accuracy reputation (printed docs) | CPU speed | Windows install | Tables / layout | Licence (code / weights) | Verdict |
|---|---|---|---|---|---|---|---|
| **RapidOCR 3.x** | PaddleOCR PP-OCR det+rec models exported to ONNX Runtime. Ships v4/v5 mobile+server and PP-OCRv6 tiny/small/medium | High (same models as PaddleOCR) | Fast for mobile models. Server/v6-medium ONNX are very slow on CPU (~90–100 s/page here) | `pip` only, pure wheels; models download once, then offline | Text only; sister projects RapidTable / RapidLayout (Apache-2.0) | Apache-2.0 / Apache-2.0 | **Tested, winner (v5 mobile)** |
| RapidOCR 1.x (`rapidocr_onnxruntime`) | Older package, PP-OCRv4 bundled in the wheel | Good | Fast | `pip`, models inside the wheel (truly offline) | – | Apache-2.0 / Apache-2.0 | Tested (old kit default) |
| **PaddleOCR 3.x** | Native PaddlePaddle runtime; PP-OCRv5 / PP-OCRv6; PP-StructureV3 for layout + tables | High; strongest open non-LLM OCR family on degraded scans in public 2026 comparisons | ~2× slower than RapidOCR here, ~2.4 GB RAM | `pip` from Paddle's CPU index; heavy (paddlex), version pins (numpy/pyyaml conflicts seen). PP-OCRv6 failed to load with paddlepaddle 3.0.0 | **PP-StructureV3** table recognition, layout, reading order | Apache-2.0 / Apache-2.0 | Tested (v5 mobile) |
| **docTR / OnnxTR** | Mindee docTR detection (DBNet/FAST) + recognition (CRNN/PARSeq/ViTSTR); OnnxTR = ONNX port, no torch | High on printed English | ~1.8–3.9 s/page; ~3.6 GB RAM | `pip install onnxtr[cpu]`; weights download once from GitHub releases | Word/line/block geometry; no table structure | Apache-2.0 / Apache-2.0 | Tested (CRNN, PARSeq) |
| **Tesseract 5** (pytesseract) | Classic LSTM engine | Great on clean 300 dpi text; weak on noise, low contrast, tables | Fast (~1.2 s/page), ~0.15 GB | Separate Windows installer (UB-Mannheim) + `pip install pytesseract` | Page segmentation modes only | Apache-2.0 / Apache-2.0 | Tested (psm 4, psm 6) |
| **EasyOCR** | CRAFT detector + CRNN, PyTorch | Mid; worst in a 22-engine 2026 comparison | Slow on CPU (~15 s/page), ~4 GB RAM | `pip`, pulls torch (~1 GB) | – | Apache-2.0 / Apache-2.0 (CRAFT weights carry MIT/NAVER terms) | Tested on 3 levels only (too slow) |
| Surya | Datalab; v2 is a ~650M-param **Qwen-style VLM** for OCR, layout and tables | High | Slow on CPU | `pip`, torch | Layout, tables, reading order | Apache-2.0 / **modified OpenRAIL-M: no commercial use above $5M revenue/funding** | Excluded: LLM-based, and its licence blocks a bank or Straive |
| MMOCR | OpenMMLab toolbox | Good (research) | Mid | Painful (mmcv builds); project largely inactive since 2023 | KIE modules | Apache-2.0 / Apache-2.0 | Excluded (maintenance, install) |
| Kraken | OCR for historical/handwritten documents | Strong on historical text | Mid | Linux/mac first; Windows unofficial | Baseline segmentation | Apache-2.0 / mixed per model | Excluded (wrong domain) |
| img2table | Table structure from lines/whitespace, wraps an OCR engine | n/a (structure only) | Fast | `pip` | **Bordered and borderless tables** | MIT | Use later for layouts the kit's header-anchor parser can't handle |
| VLM OCR (PaddleOCR-VL, olmOCR, dots.ocr, GOT-OCR, Florence-2, SmolDocling, Qwen-VL) | Vision-language models | Top of 2026 leaderboards | Need a GPU for usable speed | – | Markdown/HTML tables | Varies | Out of scope (LLM-based; CPU-impractical) |

Shortlist tested (9 configurations across 6 engines):
- RapidOCR v4 (legacy kit default), v5 mobile, v6 tiny and v6 small
- PaddleOCR v5 mobile
- OnnxTR CRNN and OnnxTR PARSeq
- Tesseract psm 4 and psm 6
- EasyOCR (3 degradation levels only)

Excluded after a speed probe: RapidOCR v5 server and v6 medium (ONNX on CPU, 87–104 s/page; the PP-OCRv6 paper's CPU speedups rely on OpenVINO). PaddleOCR native v6 small was also dropped because it failed to load with paddlepaddle 3.0.0.

## 2. Method

- **Data** (`scripts/make_dataset.py`, all fake): 12 pages, each rendered at 300 dpi and then degraded 5 ways, giving 60 page images.
  - Statement pages: 9 pages, 26 rows each, 5 fonts, in three layouts:
    - A: Date | Narration | Chq/Ref | Withdrawal | Deposit | Balance, dd/mm/yyyy
    - B: Txn Date | Description | Debit | Credit | Balance, dd-Mon-yyyy
    - C: Date | Particulars | **Amount | Dr/Cr** | Balance, dd-mm-yy
  - Amounts use Indian grouping (1,23,456.78).
  - Application forms: 3 pages with boxed fields (application ID, name, co-applicant name and ID, PAN, DOB, mobile, amounts, address).
  - Degradation levels:
    - `clean`
    - `bw200`: 200 dpi bilevel
    - `noisy_skew`: Gaussian plus salt-and-pepper noise, 1.5–3° skew, bilevel
    - `faded_blur`: blur, faded toner, 150 dpi
    - `low_contrast`: grey text, uneven lighting, slight blur
  - Ground truth: the page text line by line, plus every table cell.
- **Preprocessing variants** (kit's `preprocess.py`):
  - `raw`: no preprocessing
  - `prep`: old kit default, deskew + median filter + adaptive threshold
  - `deskew`: deskew only, grayscale kept
- **Metrics** (`scripts/score.py`), per page, averaged by engine × preprocessing × level:
  - CER/WER (jiwer) on whitespace-normalised text in reading order
  - **Amount recall**: exact-string multiset match of every ground-truth amount token; this is the critical one
  - Date recall
  - Keyword recall for withdrawal/deposit/debit/credit/balance/Dr/Cr in the header
  - **Kit reconcile**: OCR words go through `kyctrial.ocr.group_lines` and then `statements.parse_page_table` + `reconcile`. The score is the fraction of ground-truth rows whose running balance checks out (≤ ₹1).
  - Rows found, and form field exact-match
- **Speed and memory**: seconds per page (OCR only), 4 threads (`OMP/MKL` and ONNX `intra_op=4`), 8-core Linux VM with 2 jobs in parallel, so absolute times are pessimistic. Peak RSS comes from `getrusage`.

## 3. Results

All numbers are on synthetic pages. The full tables are in `results/`:
- `summary_overall.csv`
- `summary_by_level.csv`
- `summary_by_layout.csv`
- `results_per_page.csv`

### 3.1 Overall (60 pages, best preprocessing per engine = `deskew`, except where noted)

| Engine | s/page | Peak RAM | CER | Amount recall | Date recall | Reconcile (layouts A+B) | Form fields exact | Install pain |
|---|---|---|---|---|---|---|---|---|
| **RapidOCR v5 mobile** | **1.76** | **0.73 GB** | 0.007 | 0.986 | 0.993 | **0.993** | **0.960** | Low |
| PaddleOCR v5 mobile | 4.32 | 2.4 GB | **0.001** | **0.999** | 0.999 | 0.990 | 0.953 | High |
| OnnxTR (docTR) CRNN | 1.78 | 3.6 GB | 0.007 | 0.937 | 0.997 | 0.992 | 0.953 | Low |
| RapidOCR v6 tiny | **0.64** | 0.57 GB | 0.017 | 0.954 | 0.985 | 0.948 | 0.940 | Low |
| RapidOCR v6 small | 2.09 | 0.74 GB | 0.030 | 0.959 | 0.980 | 0.922 | 0.960 | Low |
| RapidOCR v4 (old default), `raw` | 2.98 | 1.9 GB | 0.081 | 0.830 | 0.948 | 0.744 | 0.827 | Lowest |
| Tesseract 5 psm 4, `prep` | 1.24 | 0.19 GB | 0.173 | 0.652 | 0.755 | 0.633 | 0.580 | Medium (separate .exe) |
| EasyOCR, `raw` (3 levels only) | 15.3 | 4.1 GB | 0.109 | 0.713 | 0.903 | 0.390 | 0.811 | Medium (torch) |

### 3.2 Amount recall by degradation level (`deskew`)

| Engine | clean | bw200 | noisy_skew | faded_blur | low_contrast |
|---|---|---|---|---|---|
| RapidOCR v5 mobile | 1.000 | 1.000 | 0.940 | 0.991 | 1.000 |
| PaddleOCR v5 mobile | 1.000 | 0.998 | 0.997 | 1.000 | 1.000 |
| OnnxTR CRNN | 0.975 | 0.940 | 0.952 | 0.891 | 0.929 |
| RapidOCR v6 tiny | 0.998 | 0.998 | 0.780 | 0.994 | 1.000 |
| RapidOCR v4 legacy (`raw`) | 0.986 | 0.782 | 0.465 | 0.980 | 0.938 |
| Tesseract psm 4 (`raw`) | 0.941 | 0.957 | 0.491 | 0.751 | 0.396 |

### 3.3 Effect of the kit's preprocessing (RapidOCR v5 mobile, all 60 pages)

| Preprocessing | Amount recall | CER | Reconcile A+B | noisy_skew reconcile |
|---|---|---|---|---|
| `raw` | 0.995 | 0.048 | 0.831 | 0.15 (skew breaks row grouping) |
| `prep` (deskew + median + threshold) | 0.735 | 0.048 | 0.866 | 0.96 |
| **`deskew` only** | 0.986 | **0.007** | **0.993** | **0.97** |

- **Binarisation hurts every neural engine.** Amount recall falls by 0.15–0.42 on `bw200` and `faded_blur`, because adaptive thresholding erodes thin digits and commas.
- Deskew is essential for table parsing: skewed rows don't group into lines.
- Bug found and fixed: the kit's skew estimate returned 0° on speckled scans, because noise breaks the text-line blobs. It now estimates on a median-filtered copy.
- Some `prep` rows in the CSV predate that fix: onnxtr_parseq, rapidocr_v4_legacy and tesseract_psm6 were not re-run, so their noisy_skew `prep` numbers are pessimistic.
- Tesseract is the exception: it does better with binarised input on low-contrast pages.

### 3.4 Layout C (single Amount + Dr/Cr column)

- Every engine reads the amounts (RapidOCR v5: 0.98 recall).
- Reconcile is only ~0.2 for all engines, because the kit's table parser has no "Amount + Dr/Cr" mode yet. That is a parser gap, not an OCR one; see the next steps.

## 4. Recommendation

1. **Default: RapidOCR 3.x, PP-OCRv5 mobile det + English rec, deskew-only preprocessing, 4 threads.** It is accurate, light (<1 GB), pure `pip` with Apache-2.0 code and weights, and needs no separate binary. This is now the kit default (`ocr.engine: rapidocr_v5`).
2. **Fallback / cross-check: PaddleOCR 3.x** if a document family beats RapidOCR. It is the same model family, and **PP-StructureV3** adds table structure for messy layouts. The cost is a heavier install and more RAM.
3. **Speed option: RapidOCR PP-OCRv6 tiny** (0.6 s/page) for a quick first pass, such as page classification only. Run v5 on the statement pages after that.
4. **Second opinion: OnnxTR.** It is a different model family, so its errors are uncorrelated. It is useful as a vote on rows that fail reconcile.
5. **Don't use** Tesseract or EasyOCR as the primary engine for these scans. Exclude Surya: its weights' licence is non-commercial above $5M revenue, and v2 is a VLM.

## 5. Caveats

- **Synthetic data is not real scans.** Real scans have stamps, signatures, handwriting, coloured logos, folds, JPEG artefacts, multi-line narrations, merged cells and vendor-specific fonts. Expect lower absolute numbers. The *ranking* is the useful output, and even that should be confirmed on the real 40 PDFs.
- The noise models are simple. Some real faults (bleed-through, dot-matrix print, carbon copies) aren't simulated.
- Times were measured on Linux under 2-way contention. On a typical 4–8 core Windows laptop, expect similar or faster times.
- I did not test on Windows itself.
  - All recommended packages ship Windows wheels.
  - RapidOCR 3.x and OnnxTR **download models on first use**. Run `python -m kyctrial.ocr` once while online, or copy the model folder, before going offline.
  - The legacy `rapidocr_onnxruntime` bundles its models in the wheel.
- Amount recall uses exact string match. The kit's `normalise_amount_text` recovers some near-misses, so the kit's own accuracy can be higher.

## 6. Suggested real-data checks (on the client laptop, data never leaves it)

1. Pick 5 PDFs (≈150 pages) covering the worst scans. Run `python -m kyctrial run` with `ocr.engine` set to `rapidocr_v5`, then `onnxtr`, then `tesseract`.
2. Compare `reconcile_rate` per file in `summary.csv`. That is a label-free accuracy signal: a row that reconciles almost certainly has correct amounts.
3. Hand-check 50 failed rows. Classify each failure as an OCR digit error, a column-assignment error or a row-merge error. This tells you whether to fix the engine or the parser.
4. Try `preprocess.denoise: true` only if the scans are heavily speckled. Keep `threshold: false` unless the scans are very low-contrast and Tesseract is used.
5. Fill in `manual_review.csv` for application ID / name / co-applicant on 20 files and run `kyctrial.evaluate`.
6. If a bank uses a single Amount + Dr/Cr column, add that parser mode first; OCR is already good enough for it.

## 7. Reproduce

```
pip install -r requirements.txt -r bench/requirements-bench.txt
cd bench
python scripts/make_dataset.py data                    # 60 PNGs (git-ignored)
python scripts/run_engine.py rapidocr_v5_mobile deskew  # writes results/raw/*.json
python scripts/score.py                                 # writes results/*.csv
```
Engine names are listed in `scripts/engines.py`.
