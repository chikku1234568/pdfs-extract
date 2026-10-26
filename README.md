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

Use 64-bit Python 3.11 or 3.12. This notebook was run on Windows with Python 3.11.9. Python 3.11.8 uses the same `cp311` wheels.

On the client laptop, use the portable Python 3.12 below. It needs no administrator account and it does not change any Python already on the machine.

Do not download the "Windows embeddable package" from python.org. That zip has no `pip` and no `venv`, and this project will not install into it.

### 1. Get the portable folder

The file is a GitHub release archive (about 40 MB), not a Hugging Face download:

https://github.com/astral-sh/python-build-standalone/releases/download/20261021/cpython-3.12.15%2B20261021-x86_64-pc-windows-msvc-install_only.tar.gz

PowerShell:

```powershell
$dest = "$env:USERPROFILE\py312"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$tar = "$env:TEMP\cpython-3.12.15-win.tar.gz"
Invoke-WebRequest -Uri "https://github.com/astral-sh/python-build-standalone/releases/download/20261021/cpython-3.12.15%2B20261021-x86_64-pc-windows-msvc-install_only.tar.gz" -OutFile $tar
tar -xzf $tar -C $dest
& "$dest\python\python.exe" --version
```

That prints `Python 3.12.15`. The program to call from here on is:

```text
%USERPROFILE%\py312\python\python.exe
```

If `Invoke-WebRequest` fails, download that URL in a browser, save the `.tar.gz`, and extract it in PowerShell:

```powershell
$dest = "$env:USERPROFILE\py312"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
tar -xzf "$env:USERPROFILE\Downloads\cpython-3.12.15+20261021-x86_64-pc-windows-msvc-install_only.tar.gz" -C $dest
& "$dest\python\python.exe" --version
```

After extract, `python.exe` must sit in `py312\python\`. The Visual C++ runtime DLLs are in that same folder, so you do not install the VC++ redistributable.

You can copy the whole `py312` folder to a USB stick or to another path. It keeps working. If you move it, use the `python.exe` inside the copied folder.

### 2. Create the project environment from that Python

Open PowerShell in the folder that contains `main.ipynb` (the clone of branch `notebook`).

The laptop firewall intercepts HTTPS, so install `pip-system-certs` before anything else. That makes pip use the Windows certificate store. RapidOCR and the other packages fail with `CERTIFICATE_VERIFY_FAILED` without it. `--trusted-host` is only for this first package, because the certificate hook does not exist yet.

```powershell
& "$env:USERPROFILE\py312\python\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pip-system-certs
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name kyc-notebook --display-name "kyc-notebook (py312)"
```

These commands call `python.exe` by path. They do not need `Activate.ps1`, and they do not need a change to the execution policy.

Check the environment before opening the notebook:

```powershell
.\.venv\Scripts\python.exe -c "import sys, paddle, rapidocr; print(sys.version); print('paddle', paddle.__version__)"
```

You want a line that starts with `3.12.15` and `paddle 3.2.0`.

### 3. Point the notebook at it

Open `main.ipynb` in VS Code or Jupyter.

- VS Code: Kernel picker, then **Select Another Kernel**, then **Python Environments**, then `.venv\Scripts\python.exe`. The label is `kyc-notebook (py312)`.
- Jupyter in the browser: Kernel menu, then **Change kernel**, then `kyc-notebook (py312)`.

Run all cells from the top. The rest of this file (weights, outputs, runtime) is the same for this Python.

### If you already have Python 3.11

Skip the portable download. In the folder that contains `main.ipynb`:

Same firewall step as above: `pip-system-certs` first, then the requirements.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pip-system-certs
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name kyc-notebook --display-name "kyc-notebook"
```

`paddlepaddle==3.2.0` is on PyPI for both 3.11 and 3.12. Do not point pip at another index.

Optional: if you prefer an activated prompt and PowerShell says "running scripts is disabled":

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

In cmd.exe the activate script is `.venv\Scripts\activate.bat`. Activation is optional. The `.\.venv\Scripts\python.exe -m ...` commands above are enough.

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
3. Select the kernel `kyc-notebook (py312)`, or `.venv\Scripts\python.exe`.
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
