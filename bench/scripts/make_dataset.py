"""Synthetic benchmark set: fake Indian bank statements (3 layouts) + application forms,
rendered at 300 dpi and degraded at 5 levels. Writes data/<deg>/<page>.png + data/gt.json.
All names/numbers are random fakes."""
import json, os, random, sys
from datetime import date, timedelta
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
W, H = 2480, 3508
# Fonts below are Debian paths; on Windows point these at C:/Windows/Fonts/arial.ttf, cour.ttf, times.ttf etc.
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/sand-box/google/Courier Prime/CourierPrime-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
FIRST = ["Ravi", "Anita", "Suresh", "Kavya", "Arjun", "Meera", "Vikram", "Lakshmi", "Rahul", "Divya"]
LAST = ["Sharma", "Reddy", "Iyer", "Nair", "Patel", "Gupta", "Rao", "Menon", "Joshi", "Kumar"]
NARR = ["UPI/{n}/PAYTM", "NEFT-{n}-SALARY", "ATM WDL {n}", "IMPS/{n}/RENT", "POS {n} AMAZON",
        "ACH D-{n} LIC", "CHQ DEP {n}", "BY CASH {n}", "INT CREDIT", "EMI {n} HDFC LTD", "SMS CHGS"]


def inr(x):  # Indian grouping 1,23,456.78
    s = f"{x:.2f}"; i, d = s.split(".")
    if len(i) > 3:
        h, t = i[:-3], i[-3:]
        g = []
        while len(h) > 2: g.insert(0, h[-2:]); h = h[:-2]
        if h: g.insert(0, h)
        i = ",".join(g) + "," + t
    return i + "." + d


def txns(rng, n, start, opening):
    d, bal, out = start, opening, []
    for _ in range(n):
        d += timedelta(days=rng.randint(0, 4))
        cr = rng.random() < 0.35
        amt = round(rng.choice([rng.uniform(10, 999), rng.uniform(1000, 99999), rng.uniform(100000, 450000)]), 2)
        if not cr and amt > bal: cr = True
        bal = round(bal + (amt if cr else -amt), 2)
        out.append(dict(date=d, narr=rng.choice(NARR).format(n=rng.randint(100000, 999999)),
                        ref=str(rng.randint(100000, 999999)) if rng.random() < .3 else "",
                        debit=None if cr else amt, credit=amt if cr else None, balance=bal))
    return out


class Page:
    def __init__(s, font, size):
        s.img = Image.new("L", (W, H), 255); s.d = ImageDraw.Draw(s.img)
        s.f = ImageFont.truetype(font, size); s.fb = ImageFont.truetype(font, size + 4)
        s.lines = []; s.cells = []
    def text(s, x, y, t, right=False, bold=False, line=True):
        f = s.fb if bold else s.f
        if right: x -= s.d.textlength(t, font=f)
        s.d.text((x, y), t, font=f, fill=0)
        if line: s.lines.append((y, x, t))
        return t


def statement(rng, layout, page_no, n_rows, start, opening, font, size):
    p = Page(font, size); lh = int(size * 1.9)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"; acct = str(rng.randint(10**11, 10**12))
    bank = ["STATE BANK OF INDIA", "HDFC BANK LTD", "ICICI BANK"][layout]
    y = 150
    if page_no == 0:
        p.text(150, y, bank, bold=True); y += lh * 2
        p.text(150, y, f"Account Name : {name}"); y += lh
        p.text(150, y, f"Account No : {acct}"); y += lh
    ts = txns(rng, n_rows, start, opening)
    fmtd = [lambda d: d.strftime("%d/%m/%Y"), lambda d: d.strftime("%d-%b-%Y"), lambda d: d.strftime("%d-%m-%y")][layout]
    if page_no == 0:
        p.text(150, y, f"Statement period from {fmtd(ts[0]['date'])} to {fmtd(ts[-1]['date'])}"); y += lh * 2
    if layout == 0:
        cols = [("Date", 150, "date", 0), ("Narration", 520, "description", 0), ("Chq/Ref No", 1180, "ref", 0),
                ("Withdrawal", 1720, "debit", 1), ("Deposit", 2040, "credit", 1), ("Balance", 2360, "balance", 1)]
    elif layout == 1:
        cols = [("Txn Date", 150, "date", 0), ("Description", 560, "description", 0),
                ("Debit", 1600, "debit", 1), ("Credit", 1980, "credit", 1), ("Balance", 2360, "balance", 1)]
    else:
        cols = [("Date", 150, "date", 0), ("Particulars", 480, "description", 0),
                ("Amount", 1780, "amount", 1), ("Dr/Cr", 1900, "drcr", 0), ("Balance", 2360, "balance", 1)]
    p.d.line((130, y - 15, 2380, y - 15), fill=0, width=3)
    hdr = []
    for t, x, k, r in cols: hdr.append(p.text(x, y, t, right=bool(r), bold=True, line=False))
    p.lines.append((y, 150, " ".join(hdr)))
    y += lh; p.d.line((130, y - 15, 2380, y - 15), fill=0, width=2)
    if page_no == 0:
        p.text(520, y, f"Opening Balance {inr(opening)}"); y += lh
    rows = []
    for t in ts:
        vals = {"date": fmtd(t["date"]), "description": t["narr"], "ref": t["ref"],
                "debit": inr(t["debit"]) if t["debit"] else "", "credit": inr(t["credit"]) if t["credit"] else "",
                "balance": inr(t["balance"]),
                "amount": inr(t["debit"] or t["credit"]), "drcr": "Dr" if t["debit"] else "Cr"}
        parts = []
        for _, x, k, r in cols:
            if vals[k]: parts.append(p.text(x, y, vals[k], right=bool(r), line=False))
        p.lines.append((y, 150, " ".join(parts)))
        rows.append({k: vals[k] for _, _, k, _ in cols}); y += lh
    p.d.line((130, y, 2380, y), fill=0, width=2); y += lh
    p.text(150, y, f"Closing Balance {inr(ts[-1]['balance'])}")
    return p, {"type": "statement", "layout": ["A_wd_dep", "B_debit_credit", "C_amount_drcr"][layout],
               "rows": rows, "opening": opening if page_no == 0 else None,
               "end_date": ts[-1]["date"], "end_bal": ts[-1]["balance"]}


def form(rng, font, size):
    p = Page(font, size); lh = int(size * 2.4); y = 200
    a = f"{rng.choice(FIRST)} {rng.choice(LAST)}"; c = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    fields = {"Application ID": f"LAP{rng.randint(10**7, 10**8)}", "Applicant Name": a,
              "Date of Birth": f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1965,2000)}",
              "PAN": "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5)) + str(rng.randint(1000, 9999)) + "F",
              "Mobile": f"9{rng.randint(10**8, 10**9)}", "Co-Applicant Name": c,
              "Co-Applicant ID": f"LAP{rng.randint(10**7, 10**8)}",
              "Loan Amount": inr(rng.randint(5, 80) * 50000), "Monthly Income": inr(rng.randint(25, 300) * 1000),
              "Address": f"{rng.randint(1,999)}, {rng.choice(['MG Road','Indiranagar','Jayanagar','HSR Layout'])}, Bengaluru 5600{rng.randint(10,99)}"}
    p.text(700, y, "LOAN APPLICATION FORM", bold=True); y += lh * 2
    for k, v in fields.items():
        p.d.rectangle((140, y - 20, 2340, y + lh - 30), outline=0, width=2)
        p.text(170, y, f"{k}", line=False); p.text(900, y, v, line=False)
        p.lines.append((y, 170, f"{k} {v}")); y += lh
    p.text(170, y + lh, "I declare that the information given above is true and correct.")
    return p, {"type": "form", "fields": fields}


def degrade(img, level, rng):
    a = np.array(img)
    r = np.random.default_rng(rng.randint(0, 10**6))
    if level == "clean":
        return a
    if level == "bw200":  # 200 dpi fax-like bilevel, resampled back to 300 for uniform input
        s = cv2.resize(a, None, fx=2/3, fy=2/3, interpolation=cv2.INTER_AREA)
        _, s = cv2.threshold(s, 150, 255, cv2.THRESH_BINARY)
        return s
    if level == "noisy_skew":
        g = a.astype(np.float32)
        g += r.normal(0, 25, g.shape); m = r.random(g.shape)
        g[m < 0.01] = 0; g[m > 0.995] = 255
        g = np.clip(g, 0, 255).astype(np.uint8)
        ang = r.uniform(1.5, 3.0) * r.choice([-1, 1])
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        g = cv2.warpAffine(g, M, (W, H), borderValue=255)
        _, g = cv2.threshold(g, 140, 255, cv2.THRESH_BINARY)
        return g
    if level == "faded_blur":
        g = cv2.GaussianBlur(a, (0, 0), 2.2)
        g = (255 - (255 - g.astype(np.float32)) * 0.55).astype(np.uint8)  # faded toner
        g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # 150 dpi
        return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    if level == "low_contrast":
        g = a.astype(np.float32) * 0.25 + 150 + r.normal(0, 6, a.shape)
        ys = np.linspace(0, 40, H)[:, None]; g = g - ys  # uneven illumination
        g = cv2.GaussianBlur(np.clip(g, 0, 255).astype(np.uint8), (0, 0), 1.0)
        return cv2.resize(cv2.resize(g, None, fx=.6, fy=.6, interpolation=cv2.INTER_AREA), (W, H))
    raise ValueError(level)


LEVELS = ["clean", "bw200", "noisy_skew", "faded_blur", "low_contrast"]


def main():
    rng = random.Random(42); pages = []
    for layout in range(3):
        for doc in range(2):
            font = FONTS[(layout * 2 + doc) % len(FONTS)]; size = rng.choice([30, 34, 38])
            start = date(2025, rng.randint(1, 6), 1); opening = round(rng.uniform(20000, 300000), 2)
            for pg in range(1 if doc else 2):
                p, gt = statement(rng, layout, pg, 26, start, opening, font, size)
                pages.append((f"st{'ABC'[layout]}{doc}p{pg}", p, gt))
                start, opening = gt["end_date"] + timedelta(days=1), gt["end_bal"]
    for i in range(3):
        p, gt = form(rng, FONTS[i], rng.choice([32, 38]))
        pages.append((f"form{i}", p, gt))
    allgt = {}
    for name, p, gt in pages:
        gt["text"] = "\n".join(t for _, _, t in sorted(p.lines))
        allgt[name] = gt
        for lv in LEVELS:
            os.makedirs(f"{OUT}/{lv}", exist_ok=True)
            cv2.imwrite(f"{OUT}/{lv}/{name}.png", degrade(p.img, lv, rng))
    json.dump(allgt, open(f"{OUT}/gt.json", "w"), default=str, indent=1)
    print(len(pages), "pages x", len(LEVELS), "levels =", len(pages) * len(LEVELS))


if __name__ == "__main__":
    main()
