"""Runs the full pipeline on the synthetic fixtures. Thresholds for the scanned
file are deliberately loose: it is a harsh 150-dpi b/w scan."""
import json
import os
import pytest
from kyctrial.config import load_config
from kyctrial.pipeline import process_pdf
from kyctrial.report import write_pdf_outputs, summary_row

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
EXP = json.load(open(os.path.join(FIX, "expected.json")))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cfg = load_config()
    res = {}
    for f in EXP:
        r = process_pdf(os.path.join(FIX, f), str(out), cfg, log=lambda *a: None)
        write_pdf_outputs(r)
        res[f] = summary_row(r)
    return res


@pytest.mark.parametrize("f", list(EXP))
def test_has_statements(results, f):
    assert results[f]["has_statements"] == EXP[f]["has_statements"]


def test_digital_exact(results):
    r, e = results["synthetic_digital.pdf"], EXP["synthetic_digital.pdf"]
    assert r["months_covered"] == e["months_covered"]
    assert r["n_transactions"] == e["n_txns"]
    assert r["reconcile_rate"] == 1.0
    for k in ("app_id", "name", "co_app_name", "co_app_id"):
        assert r[k] == e[k]


def test_scanned_loose(results):
    r, e = results["synthetic_scanned.pdf"], EXP["synthetic_scanned.pdf"]
    assert r["months_covered"] == e["months_covered"]
    assert r["n_transactions"] >= 0.6 * e["n_txns"]
    assert r["app_id"] == e["app_id"]
