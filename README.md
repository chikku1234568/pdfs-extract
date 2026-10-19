# pdfs-extract: KYC + bank-statement PDF trial

> **⚠️ Never commit client files.** Client PDFs go in `data/`, results in `outputs/`. Both are
> git-ignored and stay **on the client laptop only**. `outputs/` (OCR cache, JSON, CSV, page dumps)
> contains client text: delete it after the trial. Clear the notebook outputs before every commit.
> Run `git status` before each push. Only the synthetic PDFs in `tests/fixtures/` belong in git.

Fully offline, CPU only, open source. No LLMs, no external APIs, no network calls at runtime.
For each PDF it answers:
1. Does it contain bank statements, on which pages, and how many months?
2. Application id, applicant name, co-applicant name + id, and the statement rows
   (date, description, debit, credit, balance) with a running-balance check.

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
