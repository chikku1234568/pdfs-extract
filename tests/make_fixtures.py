"""Generate SYNTHETIC test PDFs (fake data only - Faker en_IN, fixed seed).

    python tests/make_fixtures.py   -> tests/fixtures/*.pdf + expected.json
Files:
  synthetic_digital.pdf       form + filler + 6-month statement (text layer)
  synthetic_scanned.pdf       same content, rasterised, rotated, noisy, black/white (no text layer)
  synthetic_no_statement.pdf  form + filler only, scanned style
"""
import io
import json
import os
import random
from datetime import date, timedelta

import cv2
import numpy as np
import pypdfium2 as pdfium
from faker import Faker
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")
W, H = A4
rng = random.Random(7)
fake = Faker("en_IN"); Faker.seed(7)


def application_page(c, d):
    c.setFont("Helvetica-Bold", 16); c.drawString(60, H - 70, "HOME LOAN APPLICATION FORM")
    c.setFont("Helvetica", 11)
    rows = [("Application No", d["app_id"]), ("Applicant Name", d["name"]),
            ("Date of Birth", d["dob"]), ("PAN", d["pan"]), ("Mobile", d["mobile"]),
            ("Co-Applicant Name", d["co_app_name"]), ("Co-Applicant ID", d["co_app_id"]),
            ("Loan Amount", "25,00,000"), ("Occupation", "Salaried")]
    y = H - 120
    for k, v in rows:
        c.rect(60, y - 6, 180, 22); c.rect(240, y - 6, 280, 22)
        c.drawString(66, y, k + ":"); c.drawString(246, y, v); y -= 22
    c.drawString(60, y - 30, "Address: " + d["address"][:80])
    c.drawString(60, y - 60, "Declaration: I confirm the information above is true. Signature of applicant ____")
    c.showPage()


def filler_page(c, title):
    c.setFont("Helvetica-Bold", 14); c.drawString(60, H - 70, title)
    c.setFont("Helvetica", 10)
    y = H - 100
    for _ in range(25):
        c.drawString(60, y, fake.sentence(nb_words=12)[:95]); y -= 18
    c.showPage()


def make_transactions(start, months=6, opening=54321.50):
    txns, bal, d = [], opening, start
    m = start.month - 1 + months - 1  # e.g. Apr + 5 -> Sep = 6 calendar months
    end = date(start.year + m // 12, m % 12 + 1, 28)
    while d <= end:
        if rng.random() < 0.35:
            amt = round(rng.uniform(5000, 60000), 2); bal += amt
            txns.append((d, f"NEFT CR {fake.company()[:22].upper()}", None, amt, bal))
        else:
            amt = round(rng.uniform(100, 9000), 2)
            if amt > bal:
                continue
            bal -= amt
            desc = rng.choice(["UPI/", "POS ", "ATM WDL ", "IMPS/"]) + fake.last_name().upper()
            txns.append((d, desc, amt, None, bal))
        d += timedelta(days=rng.randint(2, 6))
    return txns, opening, end


def fmt(x):
    if x is None:
        return ""
    s = f"{x:,.2f}"
    return s


def statement_pages(c, d, txns, opening, start, end, date_fmt):
    per_page = 28
    chunks = [txns[i:i + per_page] for i in range(0, len(txns), per_page)]
    xs = {"date": 50, "desc": 120, "debit": 390, "credit": 465, "balance": 550}
    for k, ch in enumerate(chunks):
        y = H - 60
        if k == 0:
            c.setFont("Helvetica-Bold", 14); c.drawString(50, y, "STATE BANK OF EXAMPLE - STATEMENT OF ACCOUNT"); y -= 20
            c.setFont("Helvetica", 10)
            c.drawString(50, y, f"Account Name: {d['name']}    Account No: XXXXXXXX{rng.randint(1000,9999)}    IFSC: SBEX0001234"); y -= 16
            c.drawString(50, y, f"Statement Period: {start.strftime('%d/%m/%Y')} to {end.strftime('%d/%m/%Y')}"); y -= 26
        c.setFont("Helvetica-Bold", 10)
        c.drawString(xs["date"], y, "Txn Date"); c.drawString(xs["desc"], y, "Particulars")
        c.drawRightString(xs["debit"], y, "Debit"); c.drawRightString(xs["credit"], y, "Credit")
        c.drawRightString(xs["balance"], y, "Balance")
        c.line(45, y - 4, 555, y - 4); y -= 18
        c.setFont("Helvetica", 9)
        if k == 0:
            c.drawString(xs["desc"], y, "Opening Balance"); c.drawRightString(xs["balance"], y, fmt(opening)); y -= 15
        for (dt, desc, dr, cr, bal) in ch:
            c.drawString(xs["date"], y, dt.strftime(date_fmt)); c.drawString(xs["desc"], y, desc)
            c.drawRightString(xs["debit"], y, fmt(dr)); c.drawRightString(xs["credit"], y, fmt(cr))
            c.drawRightString(xs["balance"], y, fmt(bal) + " Cr"); y -= 15
        if k == len(chunks) - 1:
            c.setFont("Helvetica-Bold", 9); c.drawString(xs["desc"], y - 6, "Closing Balance")
            c.drawRightString(xs["balance"], y - 6, fmt(txns[-1][4]))
        c.showPage()


def degrade(pdf_bytes, out_path, dpi=150, seed=1):
    """Rasterise each page, rotate a little, add noise/blur, binarise -> image-only PDF."""
    r = np.random.default_rng(seed)
    pdf = pdfium.PdfDocument(pdf_bytes)
    imgs = []
    for page in pdf:
        g = cv2.cvtColor(page.render(scale=dpi / 72).to_numpy()[:, :, :3], cv2.COLOR_RGB2GRAY)
        ang = r.uniform(-2.0, 2.0)
        h, w = g.shape
        g = cv2.warpAffine(g, cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1), (w, h), borderValue=255)
        g = cv2.GaussianBlur(g, (3, 3), 0.8)
        g = np.clip(g.astype(np.int16) + r.normal(0, 28, g.shape), 0, 255).astype(np.uint8)
        g = np.where(g > 150, 255, 0).astype(np.uint8)          # harsh b/w scanner threshold
        sp = r.random(g.shape)
        g[sp < 0.004] = 0                                         # specks
        imgs.append(Image.fromarray(g).convert("1"))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], resolution=dpi)
    return


def build(d, with_statement, date_fmt, start):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    application_page(c, d)
    filler_page(c, "TERMS AND CONDITIONS")
    info = {}
    if with_statement:
        txns, opening, end = make_transactions(start)
        statement_pages(c, d, txns, opening, start, end, date_fmt)
        info = {"n_txns": len(txns), "period_start": start.isoformat(), "period_end": end.isoformat(),
                "months_covered": 6}
    filler_page(c, "PROPERTY DETAILS")
    c.save()
    return buf.getvalue(), info


def person():
    return {"app_id": f"HL{rng.randint(10**7, 10**8 - 1)}", "name": fake.name(),
            "co_app_name": fake.name(), "co_app_id": f"HL{rng.randint(10**7, 10**8 - 1)}",
            "dob": fake.date_of_birth(minimum_age=25, maximum_age=60).strftime("%d/%m/%Y"),
            "pan": fake.bothify("?????####?").upper(), "mobile": "9" + str(rng.randint(10**8, 10**9 - 1)),
            "address": fake.address().replace("\n", ", ")}


def main():
    os.makedirs(OUT, exist_ok=True)
    expected = {}
    d1 = person()
    b, info = build(d1, True, "%d-%b-%Y", date(2023, 4, 3))
    open(os.path.join(OUT, "synthetic_digital.pdf"), "wb").write(b)
    expected["synthetic_digital.pdf"] = {**{k: d1[k] for k in ("app_id", "name", "co_app_name", "co_app_id")}, "has_statements": True, **info}
    d2 = person()
    b, info = build(d2, True, "%d/%m/%y", date(2024, 1, 2))
    degrade(b, os.path.join(OUT, "synthetic_scanned.pdf"), seed=2)
    expected["synthetic_scanned.pdf"] = {**{k: d2[k] for k in ("app_id", "name", "co_app_name", "co_app_id")}, "has_statements": True, **info}
    d3 = person()
    b, info = build(d3, False, "", None)
    degrade(b, os.path.join(OUT, "synthetic_no_statement.pdf"), seed=3)
    expected["synthetic_no_statement.pdf"] = {**{k: d3[k] for k in ("app_id", "name", "co_app_name", "co_app_id")}, "has_statements": False, "months_covered": 0}
    with open(os.path.join(OUT, "expected.json"), "w") as f:
        json.dump(expected, f, indent=2)
    print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
