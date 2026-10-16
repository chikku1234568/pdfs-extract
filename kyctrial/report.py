"""Step 7: write outputs - per-PDF HTML report + CSVs, and cross-PDF summary.csv."""
import json
import os
import cv2
import pandas as pd
from jinja2 import Template

THUMB_W = 700
COLORS = {"bank_statement": (0, 140, 0), "application_form": (200, 90, 0), "other": (120, 120, 120)}

HTML = Template("""<!doctype html><html><head><meta charset="utf-8"><title>{{ r.file }}</title>
<style>body{font-family:Segoe UI,Arial;margin:20px}table{border-collapse:collapse;font-size:13px}
td,th{border:1px solid #ccc;padding:3px 6px}.pg{display:inline-block;vertical-align:top;margin:6px;width:360px}
.pg img{width:350px;border:1px solid #999}.bank_statement{color:#080}.application_form{color:#c50}.other{color:#777}
.fail{background:#fdd}.ok{background:#dfd}</style></head><body>
<h2>{{ r.file }}</h2>
<p>{{ r.n_pages }} pages, processed in {{ r.seconds }} s. Contains bank statements: <b>{{ 'YES' if r.runs else 'NO' }}</b></p>
<h3>Extracted fields</h3><table><tr><th>field</th><th>value</th><th>score</th><th>page</th><th>how</th></tr>
{% for k in field_names %}{% set v = r.fields.get(k) %}<tr><td>{{k}}</td><td>{{ v.value if v else '—' }}</td>
<td>{{ v.score if v else '' }}</td><td>{{ v.page+1 if v else '' }}</td><td>{{ v.how if v else '' }}</td></tr>{% endfor %}</table>
<h3>Statement runs / months</h3><table><tr><th>run</th><th>pages</th><th>rows</th><th>distinct months</th><th>span months</th>
<th>first txn</th><th>last txn</th><th>period line</th><th>reconcile rate</th></tr>
{% for s in r.runs %}<tr><td>{{s.run}}</td><td>{{ s.pages|map('add1')|join(', ') }}</td><td>{{s.n_rows}}</td><td>{{s.distinct_months}}</td>
<td>{{s.span_months}}</td><td>{{s.start}}</td><td>{{s.end}}</td><td>{{ s.period_start ~ ' → ' ~ s.period_end if s.period_start else '—' }}</td>
<td>{{ '%.0f%%'|format(100*s.reconcile_rate) if s.reconcile_rate is not none else 'n/a' }} ({{s.rows_checked}} checked)</td></tr>{% endfor %}</table>
<h3>Pages</h3>
{% for p in r.pages %}<div class="pg"><b>Page {{p.page+1}}</b> <span class="{{p.label}}">{{p.label}}</span><br>
<small>source={{p.source}} st={{p.statement_score}} form={{p.form_score}} words={{p.n_words}} conf={{p.mean_conf}} skew={{p.skew_deg}}°
{% if p.ml_label %} ml={{p.ml_label}}({{'%.2f'|format(p.ml_prob)}}){% endif %}</small><br>
<a href="pages/p{{'%03d'|format(p.page+1)}}.png"><img src="pages/p{{'%03d'|format(p.page+1)}}.png"></a></div>{% endfor %}
<h3>Transactions (first 200)</h3><table><tr><th>run</th><th>page</th><th>date</th><th>description</th><th>debit</th><th>credit</th><th>balance</th><th>check</th></tr>
{% for t in r.transactions[:200] %}<tr class="{{t.reconciles}}"><td>{{t.run}}</td><td>{{t.page+1}}</td><td>{{t.date}}</td><td>{{t.description}}</td>
<td>{{t.debit or ''}}</td><td>{{t.credit or ''}}</td><td>{{t.balance}}</td><td>{{t.reconciles}}</td></tr>{% endfor %}</table>
<p><small>Green box = detected header row, blue box = table area, orange = field label lines.</small></p></body></html>""")

FIELD_NAMES = ["app_id", "name", "co_app_name", "co_app_id"]


def _draw(r):
    for p in r["pages"]:
        img = p.pop("_img")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        b = r["boxes"].get(p["page"], {})
        if b.get("table"):
            x0, y0, x1, y1 = map(int, b["table"]); cv2.rectangle(img, (x0, y0), (x1, y1), (255, 80, 0), 4)
        if b.get("header"):
            x0, y0, x1, y1 = map(int, b["header"]); cv2.rectangle(img, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), (0, 170, 0), 4)
        for f in r["fields"].values():
            if f["page"] == p["page"] and f.get("bbox"):
                x0, y0, x1, y1 = map(int, f["bbox"]); cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 140, 255), 3)
        cv2.putText(img, p["label"], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[p["label"]][::-1], 4)
        scale = THUMB_W / img.shape[1]
        img = cv2.resize(img, (THUMB_W, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(r["dir"], "pages", f"p{p['page']+1:03d}.png"), img)


def write_pdf_outputs(r):
    _draw(r)
    d = r["dir"]
    pd.DataFrame([{k: v for k, v in p.items() if k != "features"} | p["features"] for p in r["pages"]]) \
        .assign(page=lambda x: x["page"] + 1).to_csv(os.path.join(d, "pages.csv"), index=False)
    pd.DataFrame(r["transactions"], columns=["run", "page", "date", "description", "debit", "credit",
                                             "balance", "reconciles", "raw"]) \
        .assign(page=lambda x: x["page"] + 1).to_csv(os.path.join(d, "transactions.csv"), index=False)
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in r.items() if k not in ("pages", "transactions")}, f, indent=2, default=str)
    from jinja2 import Environment
    tpl = HTML
    tpl.environment.filters["add1"] = lambda x: x + 1
    with open(os.path.join(d, "report.html"), "w", encoding="utf-8") as f:
        f.write(tpl.render(r=r, field_names=FIELD_NAMES))


def summary_row(r):
    runs = r["runs"]
    stmt_pages = [p + 1 for s in runs for p in s["pages"]]
    checked = sum(s["rows_checked"] for s in runs)
    ok = sum((s["reconcile_rate"] or 0) * s["rows_checked"] for s in runs)
    starts = [s.get("period_start") or s["start"] for s in runs if s["start"]]
    ends = [s.get("period_end") or s["end"] for s in runs if s["end"]]
    row = {"file": r["file"], "n_pages": r["n_pages"], "has_statements": bool(runs),
           "statement_pages": " ".join(map(str, stmt_pages)), "n_runs": len(runs),
           "months_covered": max([s.get("period_months") or s["span_months"] for s in runs], default=0),
           "distinct_txn_months": max([s["distinct_months"] for s in runs], default=0),
           "period_start": min(starts) if starts else None, "period_end": max(ends) if ends else None,
           "n_transactions": len(r["transactions"]),
           "reconcile_rate": round(ok / checked, 3) if checked else None,
           "seconds": r["seconds"]}
    for f in FIELD_NAMES:
        row[f] = r["fields"].get(f, {}).get("value")
        row[f + "_score"] = r["fields"].get(f, {}).get("score")
    return row


def write_manual_review_template(summary_df, path):
    """Ground-truth sheet: copy predictions, YOU overwrite the *_true columns by hand."""
    if os.path.exists(path):
        return False  # never overwrite the user's manual work
    cols = ["file", "has_statements", "months_covered", "app_id", "name", "co_app_name", "co_app_id"]
    t = pd.DataFrame({"file": summary_df["file"]})
    for c in cols[1:]:
        t[c + "_true"] = ""
    t["notes"] = ""
    t.to_csv(path, index=False)
    return True
