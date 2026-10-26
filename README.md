# pdfs-extract

One notebook and one helper file. `main.ipynb` runs top to bottom. Functions live in `utils.py`.

Put a PDF in `data\`. The notebook renders pages, reads text with RapidOCR, labels each page with keywords (`statement`, `applicant`, or `not_needed`), reads account fields, extracts statement tables with PP-StructureV3, and writes Excel files.

This package does not call Hugging Face and does not run a language model. Keyword labels are the classifier. Step 5 of the notebook skips itself.

## Files

```
main.ipynb          run this
utils.py            helpers
requirements.txt    pip packages
data\               put PDFs here (the PDFs are not in git)
outputs\            results, created when you run (not in git)
```

## Python

Use 64-bit Python 3.11 or 3.12. This notebook was run on Windows with Python 3.11.9. Python 3.11.8 uses the same `cp311` wheels, so the install commands are the same.

Check what you already have:

```powershell
py -0p
```

If `py -3.11` or `py -3.12` prints a path, skip the portable install and use that interpreter in the venv step.

### Portable Python 3.12, no admin

Use this when the laptop has no Python, or you cannot run an installer. It is a zip-style CPython from GitHub. It does not need administrator rights and it does not change system Python. The Visual C++ runtime DLLs are inside this folder, so you do not install the VC++ redistributable.

PowerShell, any folder you can write to:

```powershell
$dest = "$env:USERPROFILE\py312"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$tar = "$env:TEMP\cpython-3.12.15-win.tar.gz"
Invoke-WebRequest -Uri "https://github.com/astral-sh/python-build-standalone/releases/download/20261021/cpython-3.12.15%2B20261021-x86_64-pc-windows-msvc-install_only.tar.gz" -OutFile $tar
tar -xzf $tar -C $dest
& "$dest\python\python.exe" --version
```

That prints `Python 3.12.15`. The interpreter is `$env:USERPROFILE\py312\python\python.exe`. You can copy the `py312` folder to another disk. It keeps working as long as you call that `python.exe`.

If `Invoke-WebRequest` fails on a TLS-inspecting proxy, download the same URL in a browser and extract the `.tar.gz` with `tar -xzf` so that `python\python.exe` sits under `py312`.

## Install

In the folder that contains `main.ipynb`:

```powershell
# existing Python 3.11:
py -3.11 -m venv .venv

# or the portable interpreter from above:
# & "$env:USERPROFILE\py312\python\python.exe" -m venv .venv

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name kyc-notebook --display-name "kyc-notebook"
```

If activation is blocked with "running scripts is disabled":

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run `Activate.ps1` again. In cmd.exe the activate script is `.venv\Scripts\activate.bat`.

Everything in `requirements.txt` is on PyPI. `paddlepaddle==3.2.0` has a Windows wheel there for Python 3.11 and 3.12. Do not point pip at another index.

If pip fails with `CERTIFICATE_VERIFY_FAILED`:

```powershell
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pip-system-certs
```

Then run the requirements install again.

## Weights (not Hugging Face)

RapidOCR weights ship inside the pip package (`rapidocr\models\`). No extra download.

PP-StructureV3 table weights are not on PyPI. The first table step downloads them from Baidu BOS:

`https://paddle-model-ecology.bj.bcebos.com`

They are OCR and table models, not a language model. The code sets these before Paddle imports, which keeps the download off Hugging Face:

```text
PADDLE_PDX_MODEL_SOURCE=bos
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
HF_HUB_OFFLINE=1
```

Files land in `%USERPROFILE%\.paddlex\official_models\`. The first download is a few hundred megabytes. Later runs reuse that folder.

The client network must allow `paddle-model-ecology.bj.bcebos.com` once. It does not need `huggingface.co`. If that host is blocked, copy an already filled `official_models` folder to the same path on the laptop.

`paddleocr` installs `huggingface-hub` as a library dependency. With the settings above, the notebook does not call it.

## Run

1. Copy the statement PDF into `data\`.
2. Open `main.ipynb`.
3. Select the kernel `kyc-notebook` (or the `.venv` interpreter).
4. Run all cells from the top.

If `data\` has one PDF, the first cell uses it. If it has several, the cell prints the list and uses the first. Set `PDF = Path("data/your_file.pdf")` to pick another.

From the activated venv, without the notebook UI, you can also run the same cells with Jupyter:

```powershell
python -m pip install nbclient
jupyter nbconvert --to notebook --execute main.ipynb --inplace
```

Close other heavy programs first. PP-StructureV3 on CPU needs about 7 GB of RAM and about 30–90 seconds per page. These pages are large (about 6900×9700 pixels at 200 DPI), so a 6-page file can take longer. `enable_mkldnn=False` is set in `utils.py`. Leave it. MKL-DNN used much more RAM.

Cell 9 writes `outputs\<pdf-name>\ppstruct_pN.json`. A later run skips pages that already have that file. Delete the file to parse that page again.

## Output

For a PDF named `statement.pdf`:

| File | What it is |
|---|---|
| `outputs\statement\ocr_p1.json` | OCR boxes for page 1 (one file per page) |
| `outputs\statement\ppstruct_p1.json` | table structure for page 1 |
| `outputs\statement\transactions.xlsx` | date, description, debit/credit, amount, balance |
| `outputs\statement\leads.xlsx` | customer id, balances, month flags |

Month flags: full means 6 or more distinct months, partial means 1–5, no banking data means 0.

## Windows run of `data\00023.pdf`

Recorded on this PC (Python 3.11.9, CPU). The PDF is not in git.

- 6 pages, all labelled `statement`
- Customer ID `460800283`
- Opening balance `1989440.85`, closing balance `3789730.40`
- Account `62045674489`, name `FASHION TEXTILES LTD`
- Period January–March 2024, so 3 months, flag **partial**
- 129 transactions

OCR took about 25 seconds. The table step took about 90 seconds per page after the weights were on disk. The first table run also downloaded the PP-Structure weights from Baidu BOS. A second run of those pages is faster because `ppstruct_pN.json` is reused.

## If import fails

`DLL load failed` on `paddle`, `onnxruntime`, or `cv2`: use the portable Python 3.12 folder above. Its `python.exe` directory already contains `vcruntime140.dll`. The system-wide VC++ installer needs admin, so skip it when you do not have admin rights.
