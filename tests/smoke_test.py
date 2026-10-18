"""Quick check that the pipeline still works on the synthetic fixtures (fake data).

    python tests/smoke_test.py            # RapidOCR venv
    python tests/smoke_test.py paddle     # PaddleOCR venv

Prints PASS/FAIL per file and field vs tests/fixtures/expected.json. The scanned fixture is
deliberately awful, so a few row-level misses are normal; the checks below are the coarse ones."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import common  # noqa: E402

engine = sys.argv[1] if len(sys.argv) > 1 else "rapid"
if engine == "rapid":
    from utils.ocr_rapid import ocr_page
else:
    from utils.ocr_paddle import ocr_page

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
common.CACHE_DIR = None
expected = json.load(open(os.path.join(here, "expected.json"), encoding="utf-8"))
fails = 0
for fname, exp in expected.items():
    s = common.process_pdf(os.path.join(here, fname), ocr_page, engine, log=None)["summary"]
    checks = {"has_statements": s["has_statements"] == exp["has_statements"],
              "months_covered": s["months_covered"] == exp["months_covered"]}
    for f in common.FIELD_NAMES:
        checks[f] = s[f] == exp[f]
    if exp["has_statements"]:
        checks["n_transactions>=80%"] = s["n_transactions"] >= 0.8 * exp["n_txns"]
    for k, ok in checks.items():
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {fname:28s} {k:20s} got={s.get(k.split('>')[0])!r}")
    print(f"      {fname:28s} balance_ok_rate={s['balance_ok_rate']}  {s['seconds']}s")
print("ALL PASS" if not fails else f"{fails} FAIL(S)")
sys.exit(1 if fails else 0)
