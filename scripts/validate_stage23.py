"""Validate the Stage 1->2->3 cascade + smooth_promote on synthetic pages.
Run from repo root: python scripts/validate_stage23.py [--pdf data/synthetic_classify.pdf]
Uses the same path as main.ipynb: pypdfium2 render -> gray -> RapidOCR -> lines -> classify_page -> smooth_promote.
Model download/load failures are reported per page, not fatal. Exit code 1 only if an assertion fails."""
import sys, time, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import utils as U
import make_synthetic_classify_pages as G

def test_smooth_promote():
    cases = [(["statement", "not_needed", "statement"], ["statement"] * 3),
             (["applicant", "not_needed", "statement"], ["applicant", "applicant", "statement"]),
             (["not_needed", "statement", "not_needed"], ["not_needed", "statement", "not_needed"]),
             (["statement", "not_needed", "not_needed", "statement"], ["statement", "not_needed", "not_needed", "statement"]),
             (["not_needed"], ["not_needed"]), ([], [])]
    return [(c, U.smooth_promote(c) == e) for c, e in cases]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pdf", default=str(ROOT / "data/synthetic_classify.pdf"))
    a = ap.parse_args()
    if not Path(a.pdf).exists(): G.build(a.pdf)
    pages = U.render_pages(a.pdf, dpi=200)
    rows, t_by = [], {}
    for i in sorted(pages):
        img = pages[i]; t0 = time.time()
        text = "\n".join(U.lines_from_words(U.ocr_page(U.to_gray(img))))
        t_ocr = time.time() - t0; t0 = time.time()
        try: lab, st, info = U.classify_page(text, img)
        except Exception as e: lab, st, info = "ERROR", "error", {"error": repr(e)[:200]}
        dt = time.time() - t0; t_by.setdefault(st, []).append(dt)
        rows.append(dict(page=i + 1, raw=lab, stage=st, t_ocr=t_ocr, t_cls=dt, info=info))
    final = U.smooth_promote([r["raw"] for r in rows])
    print(f"\n{'pg':>2} {'expected':<11}{'raw':<11}{'final':<11}{'stage':<18}{'cls_s':>6}  signals")
    for r, f, (_, exp, exp_st, note) in zip(rows, final, G.PAGES):
        r["final"] = f; inf = r["info"]
        sig = {"s1": inf.get("s1")}
        for k in ("s2_layout", "s2_zs", "s3_layout", "s3_zs", "s3_rvl", "error"):
            if k in inf: sig[k] = inf[k]
        ok = "OK " if f == exp else "XX "
        print(f"{r['page']:>2} {exp:<11}{r['raw']:<11}{f:<11}{r['stage']:<18}{r['t_cls']:6.1f}  {ok}{note}\n     {json.dumps(sig)}")
    print("\nTiming by decided stage (s/page):", {k: round(sum(v) / len(v), 2) for k, v in t_by.items()})
    st = [r["stage"] for r in rows]
    checks = {
        "Stage2 keep exercised": any(s.startswith("S2") for s in st),
        "Stage3 reached": any(s.startswith("S3") for s in st),
        "Stage3 confirmed drop": "S3_confirmed_drop" in st,
        "applicant keep (S1/S2)": any(r["raw"] == "applicant" and r["stage"][:2] in ("S1", "S2") for r in rows),
        "smooth_promote promoted a sandwiched page in PDF": any(r["raw"] == "not_needed" and r["final"] != "not_needed" for r in rows),
        "smooth_promote unit tests": all(ok for _, ok in test_smooth_promote()),
        "no model/page errors": not any(r["stage"] == "error" for r in rows),
    }
    match = sum(r["final"] == p[1] for r, p in zip(rows, G.PAGES))
    print(f"\nExpected-label match: {match}/{len(rows)} (informational; models may shift)")
    for k, v in checks.items(): print(("PASS " if v else "FAIL ") + k)
    hard = [k for k, v in checks.items() if not v and k != "no model/page errors"]
    sys.exit(1 if hard else 0)

if __name__ == "__main__":
    main()
