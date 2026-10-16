from kyctrial.parsing import parse_amount, parse_date, parse_date_lenient
from kyctrial.statements import group_runs, months_covered, reconcile
from datetime import date


def test_dates():
    assert parse_date("03/04/2023") == date(2023, 4, 3)
    assert parse_date("03-04-23") == date(2023, 4, 3)
    assert parse_date("03-Apr-2023") == date(2023, 4, 3)
    assert parse_date_lenient("260124") == date(2024, 1, 26)


def test_amounts():
    assert parse_amount("1,23,456.78 Cr") == (123456.78, "cr")
    assert parse_amount("500.00Dr") == (500.0, "dr")
    assert parse_amount("117.886.29") == (117886.29, None)   # OCR dots
    assert parse_amount("2023") == (None, None)               # years/ids are not amounts


def test_runs_and_months():
    assert group_runs(["other", "bank_statement", "bank_statement", "other", "bank_statement"], 1) == [[1, 2, 4]]
    m = months_covered([date(2023, 4, 3), date(2023, 9, 28), date(2023, 6, 1)])
    assert m["span_months"] == 6 and m["distinct_months"] == 3


def test_reconcile():
    rows = [{"debit": None, "credit": 100.0, "balance": 200.0},
            {"debit": 50.0, "credit": None, "balance": 150.0},
            {"debit": 10.0, "credit": None, "balance": 999.0}]
    rate, n = reconcile(rows, opening=100.0)
    assert n == 3 and abs(rate - 2 / 3) < 1e-9
