"""Glue: run every step for one PDF and return a result dict."""
import os
import time
import cv2

from . import render, preprocess, ocr, classify, statements, fields


def process_pdf(pdf_path, out_dir, cfg, max_pages=None, debug=False, ml_model=None, log=print):
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdir = os.path.join(out_dir, name)
    os.makedirs(os.path.join(pdir, "pages"), exist_ok=True)
    cache_dir = os.path.join(out_dir, "_ocr_cache")
    n = render.page_count(pdf_path)
    if max_pages:
        n = min(n, max_pages)

    pages, page_lines = [], {}
    t_all = time.time()
    for p in range(n):
        t0 = time.time()
        pg = render.load_page(pdf_path, p, cfg)
        angle, cached = 0.0, None
        if pg["source"] == "text":
            words, show = pg["words"], pg["image"]
        else:
            dbg = os.path.join(pdir, "debug") if debug else None
            clean, angle = preprocess.preprocess(pg["image"], cfg, dbg, tag=f"p{p+1:03d}")
            show = clean
            words, cached = ocr.ocr_cached(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR) if clean.ndim == 2 else clean,
                                           pdf_path, p, cfg, cache_dir)
        cls, feats = classify.classify_page(words, cfg)
        if ml_model is not None:
            cls["ml_label"], cls["ml_prob"] = classify.predict_ml(ml_model, feats["text"])
        page_lines[p] = feats["lines"]
        dt = time.time() - t0
        pages.append({"page": p, "source": pg["source"], "skew_deg": round(angle, 2),
                      "ocr_cached": cached, "n_words": len(words),
                      "mean_conf": round(sum(w["conf"] for w in words) / len(words), 3) if words else None,
                      "seconds": round(dt, 2), "text": feats["text"], **cls, "_img": show})
        log(f"  page {p+1}/{n}: {pg['source']:5s} {cls['label']:16s} st={cls['statement_score']:.2f} "
            f"form={cls['form_score']:.2f} words={len(words)} {dt:.1f}s{' (cache)' if cached else ''}")

    # statements
    runs = statements.group_runs([pg["label"] for pg in pages], cfg["statements"]["max_page_gap"])
    run_results, txns, boxes = [], [], {}
    for k, run in enumerate(runs):
        summ, rows, bx = statements.analyse_run(run, page_lines, cfg)
        summ["run"] = k + 1
        for r in rows:
            r["run"] = k + 1
        run_results.append(summ); txns += rows; boxes.update(bx)

    # fields: search non-statement pages first, keep best score per field
    found = {}
    order = [p for p in range(n) if pages[p]["label"] != "bank_statement"] + \
            [p for p in range(n) if pages[p]["label"] == "bank_statement"]
    for p in order:
        for f, v in fields.extract_fields(page_lines[p], cfg, page_no=p).items():
            if pages[p]["label"] == "bank_statement":
                v["score"] *= 0.5  # names on statements are account holders, weaker evidence
            if f not in found or v["score"] > found[f]["score"]:
                found[f] = v

    return {"file": os.path.basename(pdf_path), "name": name, "dir": pdir, "n_pages": n,
            "pages": pages, "runs": run_results, "transactions": txns, "boxes": boxes,
            "fields": found, "seconds": round(time.time() - t_all, 1)}
