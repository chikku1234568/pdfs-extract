"""Command line entry point.

    python -m kyctrial run --input data/ --output outputs/ [--dpi 250] [--max-pages 5] [--debug]
    python -m kyctrial train --labels labels.csv --model outputs/page_clf.joblib
"""
import argparse
import glob
import os
import sys
import time

import pandas as pd

from .config import load_config


def cmd_run(a):
    cfg = load_config(a.config)
    if a.dpi:
        cfg["dpi"] = a.dpi
    from .pipeline import process_pdf
    from .report import write_pdf_outputs, summary_row, write_manual_review_template
    model = None
    if a.model:
        import joblib
        model = joblib.load(a.model)
    pdfs = sorted(glob.glob(os.path.join(a.input, "**", "*.pdf"), recursive=True)) if os.path.isdir(a.input) else [a.input]
    if not pdfs:
        sys.exit(f"No PDFs found in {a.input}")
    os.makedirs(a.output, exist_ok=True)
    rows = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf}")
        try:
            r = process_pdf(pdf, a.output, cfg, a.max_pages, a.debug, model)
        except Exception as e:  # keep going; one bad file must not kill the batch
            print(f"  ERROR: {type(e).__name__}: {e}  (password-protected or corrupt?)")
            rows.append({"file": os.path.basename(pdf), "error": f"{type(e).__name__}: {e}"})
            continue
        write_pdf_outputs(r)
        rows.append(summary_row(r))
        print(f"  -> {r['dir']}/report.html  ({r['seconds']}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.output, "summary.csv"), index=False)
    if write_manual_review_template(df, os.path.join(a.output, "manual_review.csv")):
        print("Wrote manual_review.csv template - fill the *_true columns, then run kyctrial.evaluate")
    print(f"Done. Summary: {os.path.join(a.output, 'summary.csv')}")


def cmd_train(a):
    from .classify import train_ml
    train_ml(a.labels, a.model)
    print("Saved", a.model)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kyctrial")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a folder (or one file) of PDFs")
    r.add_argument("--input", required=True)
    r.add_argument("--output", default="outputs")
    r.add_argument("--dpi", type=int, help="render DPI for scanned pages (default 250)")
    r.add_argument("--max-pages", type=int, help="only first N pages per PDF (quick trials)")
    r.add_argument("--debug", action="store_true", help="save preprocessing step images")
    r.add_argument("--config", help="YAML file overriding config.py defaults")
    r.add_argument("--model", help="optional trained page classifier (.joblib)")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("train", help="train TF-IDF+LogReg page classifier from labels CSV (text,label)")
    t.add_argument("--labels", required=True)
    t.add_argument("--model", default="outputs/page_clf.joblib")
    t.set_defaults(fn=cmd_train)
    a = ap.parse_args(argv)
    a.fn(a)
