# pdfs-extract — notebook-first trial (branch `dev`)

**Main path:** `main.ipynb` + `utils.py` (functions only). Older scripts are in `legacy/`.
Pipeline: ingest (pypdfium2) → grayscale/deskew (OpenCV) → OCR (RapidOCR) → page class (`applicant` / `statement` / `not_needed`) → applicant fields (Customer ID, opening/closing balance, account, name, period) → tables (**PP-StructureV3**) → normalised transactions → months → lead flags → `outputs\<lead>\transactions.xlsx` and `leads.xlsx`.

**Page classification:** keyword scoring is the **primary** classifier and stays as-is. BERT (DistilBERT from **ModelScope, not Hugging Face**) is an **optional fallback only**, used when keywords return `not_needed`. You can skip installing it.

## Run on Windows (primary guide)

> Developed and smoke-tested on **Linux**; the **target runtime is Windows**. Differences that matter: use the `py` launcher (not `python3`), activate with `.venv-paddle\Scripts\Activate.ps1` (not `source .../bin/activate`), and use backslash paths (`data\file.pdf`). Paths inside the notebook are built with `pathlib`, so they work on both.

**Prerequisites**
- Python **3.12** 64-bit (python.org installer, tick "py launcher"). PaddlePaddle has no wheels for 3.14; 3.12 is the tested version.
- **Microsoft Visual C++ Redistributable (x64)** — install it if `import paddle`, `onnxruntime` or `cv2` fails with a DLL load error: https://aka.ms/vs/17/release/vc_redist.x64.exe
- VS Code (Python + Jupyter extensions) or Jupyter.

**Setup (PowerShell, in the repo folder)** — one venv, `.venv-paddle`, for the whole notebook:
```powershell
py -3.12 -m venv .venv-paddle
.venv-paddle\Scripts\Activate.ps1
# If blocked ("running scripts is disabled"):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    (then re-run Activate.ps1)
python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name kyc-paddle --display-name "kyc-paddle (Py3.12)"
```
(cmd.exe instead of PowerShell: `.venv-paddle\Scripts\activate.bat`.)

**Optional BERT fallback** (skip unless you want to compare):
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers modelscope
modelscope download --model AI-ModelScope/distilbert-base-uncased --include "*.safetensors" "*.json" "vocab.txt" --local_dir models\distilbert-base-uncased
```

**SSL errors on a corporate proxy** (`CERTIFICATE_VERIFY_FAILED` from pip or model downloads):
```powershell
python -m pip install pip-system-certs
# if pip itself can't connect:
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pip-system-certs
```
Then retry. For the Paddle index also add `--trusted-host www.paddlepaddle.org.cn` if needed.

**Open and run**
- VS Code: open `main.ipynb` → *Select Kernel* → `kyc-paddle (Py3.12)` (or the `.venv-paddle` interpreter).
- Or: `jupyter notebook main.ipynb` from the activated venv and pick kernel `kyc-paddle`.
- Put PDFs in `data\` and set `PDF = ...` in the first code cell. `data\`, `outputs\`, `models\` are gitignored.
- Run top to bottom. Cell 9 caches PP-StructureV3 results per page as JSON, so re-runs skip finished pages.

**Where models land on Windows** (copy these to an offline machine at the same location):
- RapidOCR: inside site-packages, `.venv-paddle\Lib\site-packages\rapidocr\models\`. Exact path:
  `python -c "import rapidocr, os; print(os.path.join(os.path.dirname(rapidocr.__file__), 'models'))"`
- PP-StructureV3: `%USERPROFILE%\.paddlex\official_models\` (downloaded once from Baidu BOS, not Hugging Face). Set env var `PADDLE_PDX_CACHE_HOME` to move it.
- Optional DistilBERT: `models\distilbert-base-uncased\` in the repo.

**RAM / speed**
- PP-StructureV3 on CPU: ~25–90 s per page, **~7 GB peak RAM**. Close browsers/other heavy apps on a 16 GB laptop.
- `enable_mkldnn=False` is already set in the code — leave it; MKL-DNN on caused an out-of-memory crash (~12 GB).

**Two venvs?** Not needed. An older `.venv-rapid` can run the OCR/classification/applicant cells, but the table cells need Paddle, so use `.venv-paddle` for everything.

**Dummy PDF result (Linux box):** all 6 pages `statement`; Customer ID 460800283, opening 1,989,440.85, closing 3,789,730.40; 129 of 157 transactions; 3 months → **Partial**.

---

# LEGACY (old script pipeline, `legacy/`)

> **Legacy.** Everything below describes the old script layout (`utils/common.py`, `utils/ocr_rapid.py`, `main_rapid.py`, ...), now moved to `legacy/`. Paths below are **not** valid for `main.ipynb`. Use the Windows guide at the top.

## (legacy) pdfs-extract: KYC + bank-statement PDF trial

## Layout

```
data/                 put PDFs here (git-ignored)
outputs/              results (git-ignored)
utils/common.py       everything except OCR: settings at the top, then render, deskew,
                      classify, statement table + balance check + months, fields, run_folder, evaluate
utils/ocr_rapid.py    RapidOCR PP-OCRv5 mobile   -> ocr_page(img) -> [{text,x0,y0,x1,y1,conf}]
utils/ocr_paddle.py   PaddleOCR PP-OCRv5 mobile  -> same ocr_page(img)
main_rapid.py         run every PDF in data/ with RapidOCR
main_paddle.py        same with PaddleOCR
main.ipynb            step by step: render, OCR, words, classify, table, fields, whole folder
dump_page.py          one page -> outputs/dump_<pdf>_p<N>.json (words, lines, classifier scores)
tests/                synthetic fixtures + smoke_test.py
bench/                OCR engine benchmark (see bench/OCR_RESEARCH.md)
docs/                 questions for the client, next steps, results on fixtures
```

Flow per page: text layer present? use it : render at `DPI` → grayscale + deskew → OCR (cached in
`outputs/_ocr_cache/`) → group into lines → score `st` (statement) / `form` → label. Consecutive
statement pages form one statement: header row (credit/debit/balance) sets the columns, rows get
`prev + credit − debit == balance`, months come from the "period … to …" line or the transaction dates.
Fields: fuzzy label match, value to the right or below, regex check.

Everything you would tune is a constant at the top of `utils/common.py`: `DPI`, `DESKEW`, `BINARISE`
(keep off), `STATEMENT_MIN_SCORE`, keyword lists, `COLUMN_SYNONYMS` (bank header names), `FIELDS`
(label synonyms + regex). OCR threads / min confidence are at the top of each `utils/ocr_*.py`.

## Setup on Windows (two venvs)

Python 64-bit (RapidOCR works on 3.11–3.14; PaddleOCR needs 3.9–3.13, as paddlepaddle has no 3.14 wheel yet, so build its venv with `py -3.12`), plus the Microsoft Visual C++ Redistributable (x64). In the VS Code
terminal (PowerShell), in the repo folder:

```powershell
# RapidOCR venv
python -m venv .venv-rapid
.venv-rapid\Scripts\Activate.ps1       # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-rapid.txt
python utils\ocr_rapid.py               # one-time model download (needs internet)
deactivate

# PaddleOCR venv
py -3.12 -m venv .venv-paddle
.venv-paddle\Scripts\Activate.ps1
pip install -r requirements-paddle.txt
python utils\ocr_paddle.py              # one-time model download
deactivate
```

**SSL errors** (corporate network inspecting HTTPS): `pip install pip-system-certs` in that venv,
then retry. If pip itself fails:
`pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pip-system-certs`.

**Where the models live** (copy these folders to an offline machine, same place):
- RapidOCR: `.venv-rapid\Lib\site-packages\rapidocr\models\` (3 `.onnx` files). Exact path:
  `python -c "import rapidocr, os; print(os.path.join(os.path.dirname(rapidocr.__file__), 'models'))"`
- PaddleOCR: `%USERPROFILE%\.paddlex\official_models\` (`PP-OCRv5_mobile_det`, `en_PP-OCRv5_mobile_rec`).
  Set env var `PADDLE_PDX_CACHE_HOME` to use another folder.

In VS Code pick the kernel/interpreter from `.venv-rapid` or `.venv-paddle` (Python: Select Interpreter).

## Run

```powershell
.venv-rapid\Scripts\Activate.ps1
python tests\smoke_test.py                       # fixtures, should print ALL PASS
python main_rapid.py                             # every PDF in data\
python main_rapid.py --max-pages 5 --dpi 300     # quick look / tuning
python main_rapid.py --data tests\fixtures       # synthetic files
python dump_page.py data\file01.pdf 5            # what OCR read on page 5
```

Same with PaddleOCR in its venv: `python tests\smoke_test.py paddle`, `python main_paddle.py`,
`python dump_page.py data\file01.pdf 5 --engine paddle`.

Outputs (per engine, so you can compare them side by side):

| File | What |
|---|---|
| `outputs/<engine>/summary.csv` | one row per PDF: `has_statements, statement_pages, months_covered, distinct_txn_months, period_start/end, n_transactions, balance_ok_rate, app_id, name, co_app_name, co_app_id` (+ scores) |
| `outputs/<engine>/<pdf>.json` | everything: fields, statements, transactions, pages (label, `st`/`form` scores, features, text) |
| `outputs/<engine>/<pdf>_transactions.csv` | statement rows, `balance_ok` = ok / fail / na |
| `outputs/manual_review.csv` | fill the `*_true` columns by hand (never overwritten) |
| `outputs/dump_<pdf>_p<N>.json` | from `dump_page.py`: source, words with bbox + conf, lines, label, scores, features |

- `months_covered`: from the statement period line if found, else first to last transaction month.
- `balance_ok_rate`: share of rows where previous balance + credit − debit = balance (±₹1). Low = OCR
  digit errors or wrong columns: look at the `raw` text of the `fail` rows.
- Rerunning is fast: OCR is cached per page in `outputs/_ocr_cache/` (`--no-cache` to redo it).

## Notebook

Open `main.ipynb`, choose the kernel of the venv you want, set `ENGINE = "rapid"` or `"paddle"`
and run cell by cell. Edit `utils/common.py` and re-run the reload cell to try changes.
Accuracy: fill `outputs/manual_review.csv`, then the last cell runs `common.evaluate(...)`.
**Clear All Outputs before committing.**

Docs: [questions for the client](docs/QUESTIONS_FOR_CLIENT.md), [next steps](docs/NEXT_STEPS.md),
[results on the fixtures](docs/TRIAL_RESULTS.md), [OCR benchmark](bench/OCR_RESEARCH.md).
Synthetic fixtures can be regenerated with `python tests\make_fixtures.py` (needs `pip install reportlab==5.0.1 Faker==40.43.0`).
