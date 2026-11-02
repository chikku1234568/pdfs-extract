"""Generate data/synthetic_classify.pdf: synthetic (fake) pages that exercise Stage 1/2/3 + smooth_promote.
Run: python scripts/make_synthetic_classify_pages.py   (needs Pillow only). No real data.
EXPECTED maps page -> (expected label, expected stage path) used by validate_stage23.py."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

W, H = 1654, 2339  # A4 @200dpi
FD = "/usr/share/fonts/truetype/dejavu/"
def F(sz, bold=False):
    for p in [FD + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "arial.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except OSError: pass
    return ImageFont.load_default()

def page():
    im = Image.new("RGB", (W, H), "white"); return im, ImageDraw.Draw(im)

def para(d, y, text, sz=30, x=120, width=78):
    import textwrap
    for ln in textwrap.wrap(text, width):
        d.text((x, y), ln, fill="black", font=F(sz)); y += int(sz * 1.5)
    return y + 20

def strong_statement(seed):
    random.seed(seed); im, d = page()
    d.text((120, 100), "XYZ BANK - STATEMENT OF ACCOUNT", fill="black", font=F(44, True))
    d.text((120, 180), "Account No: 001122334455   Period: 01/04/2025 - 30/04/2025", fill="black", font=F(28))
    cols = [120, 330, 820, 1060, 1300]; hdr = ["Date", "Narration", "Debit", "Credit", "Balance"]
    y = 280
    for x, h in zip(cols, hdr): d.text((x, y), h, fill="black", font=F(30, True))
    bal = 50000.0
    for i in range(18):
        y += 70; amt = round(random.uniform(100, 9000), 2); dr = random.random() < .6
        bal += -amt if dr else amt
        row = [f"{i+1:02d}/04/2025", random.choice(["UPI/GROCERY", "NEFT SALARY", "ATM WDL", "IMPS TRF"]),
               f"{amt:,.2f}" if dr else "", "" if dr else f"{amt:,.2f}", f"{bal:,.2f}"]
        for x, v in zip(cols, row): d.text((x, y), v, fill="black", font=F(28))
    return im

def weak_statement():
    """Ruled grid table, only 2 dated rows, <4 finance headers -> Stage1 miss, Stage2 table keep."""
    im, d = page()
    d.text((120, 120), "Summary Sheet", fill="black", font=F(44, True))
    xs = [120, 520, 1000, 1500]; ys = [300 + 90 * k for k in range(9)]
    for x in xs: d.line([(x, ys[0]), (x, ys[-1])], fill="black", width=3)
    for y in ys: d.line([(xs[0], y), (xs[-1], y)], fill="black", width=3)
    cells = [["Particulars", "Ref", "Value"], ["Opening", "A-17", "12,400.00"], ["Grocery 03/05/2025", "B-22", "1,250.00"],
             ["Fuel 09/05/2025", "C-31", "2,100.50"], ["Rent", "D-04", "18,000"], ["Utilities", "E-11", "3,420"],
             ["Misc", "F-90", "760"], ["Closing", "G-01", "6,869.50"]]
    for r, row in enumerate(cells):
        for c, v in enumerate(row): d.text((xs[c] + 20, ys[r] + 25), v, fill="black", font=F(30, r == 0))
    return im

def strong_applicant():
    im, d = page()
    d.text((120, 100), "LOAN APPLICATION FORM", fill="black", font=F(46, True))
    fields = [("Application ID", "APP-2025-778812"), ("Customer ID", "CUST99812"), ("Applicant Name", "Ravi Kumar Test"),
              ("PAN", "ABCDE1234F"), ("Aadhaar", "XXXX-XXXX-1234"), ("Date of Birth", "12/08/1990"),
              ("Address", "12 Sample Street, Testnagar, Pune 411001"), ("Mobile", "98XXXXXX10"),
              ("Occupation", "Salaried"), ("Annual Income", "6,00,000")]
    y = 230
    for k, v in fields:
        d.rectangle([110, y - 10, 1540, y + 60], outline="black", width=2)
        d.text((130, y), k, fill="black", font=F(30, True)); d.text((620, y), v, fill="black", font=F(30)); y += 90
    d.rectangle([1200, 1300, 1500, 1650], outline="black", width=3); d.text((1255, 1460), "Photograph", fill="black", font=F(28))
    d.text((130, 1750), "Signature: ____________________", fill="black", font=F(30))
    return im

def weak_applicant():
    """Sparse personal-details form with a round seal; no IDs, <3 cue words -> Stage1 miss."""
    im, d = page()
    d.text((120, 120), "Personal Details", fill="black", font=F(46, True))
    y = 280
    for k in ["Full Name", "Residence", "Phone", "Place of Work", "Monthly Earnings", "Guarantor"]:
        d.text((130, y), k + ":", fill="black", font=F(32, True)); d.line([(560, y + 40), (1500, y + 40)], fill="black", width=2)
        y += 120
    cx, cy, r = 1250, 1500, 180
    for k in range(3): d.ellipse([cx - r + k * 6, cy - r + k * 6, cx + r - k * 6, cy + r - k * 6], outline=(30, 30, 160), width=4)
    d.text((cx - 120, cy - 20), "BRANCH SEAL", fill=(30, 30, 160), font=F(34, True))
    d.text((130, 1900), "Verified by officer", fill="black", font=F(30))
    return im

TC = ("These terms govern use of the services described herein. By continuing you agree to be bound by these terms. "
      "We may revise these terms at any time without notice. Any dispute shall be subject to the exclusive jurisdiction "
      "of the courts at the registered office. Liability is limited to the extent permitted by law. ")
PRIV = ("This privacy notice explains how we collect, store and process personal information. Information may be "
        "shared with service providers under contract. You may request access or correction by writing to us. "
        "Cookies are used to improve the website experience. ")
MKT = ("Festive Offer! Get flat 20 percent off on shopping with our rewards program. Earn points on every purchase. "
       "Limited period only. Visit our website or nearest store today. Terms apply. ")

def junk(title, body, reps=6):
    im, d = page(); d.text((120, 120), title, fill="black", font=F(46, True)); y = 260
    for k in range(reps): y = para(d, y, f"{k+1}. " + body)
    return im

def blankish():
    im, d = page(); d.text((W // 2 - 200, H // 2), "This page is intentionally left blank", fill=(90, 90, 90), font=F(30)); return im

# (builder, expected_label, expected_stage_prefix, note)
PAGES = [
    (lambda: strong_statement(1), "statement", "S1", "strong statement"),
    (blankish, "statement", "S3_drop+smooth", "blank sandwiched -> smooth_promote"),
    (lambda: strong_statement(2), "statement", "S1", "strong statement"),
    (weak_statement, "statement", "S2", "weak statement: ruled table, 2 dated rows"),
    (strong_applicant, "applicant", "S1", "applicant form with App ID/PAN/Aadhaar"),
    (weak_applicant, "applicant", "S2|S3", "weak applicant: sparse fields + seal"),
    (lambda: junk("Privacy Notice", PRIV), "not_needed", "S3_confirmed_drop", "privacy policy"),
    (lambda: junk("Festive Offer", MKT, 4), "not_needed", "S3_confirmed_drop", "marketing flyer"),
    (lambda: junk("Terms and Conditions", TC), "not_needed", "S3_confirmed_drop", "T&C prose"),
]

def build(out="data/synthetic_classify.pdf"):
    ims = [b() for b, *_ in PAGES]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ims[0].save(out, save_all=True, append_images=ims[1:], resolution=200)
    return out

if __name__ == "__main__":
    print(build())
