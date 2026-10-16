# Questions for the client / what the trial should answer

## Data & formats
1. Which banks appear (list), and roughly how many distinct statement layouts? Can we get one **redacted** sample page per layout to tune header synonyms on-site?
2. Are statements always tables with Credit / Debit / Balance columns, or also single "Amount" + Dr/Cr columns, passbook photocopies, or e-statement printouts?
3. Can one PDF contain **several accounts** or several applicants' statements? Should months be counted per account, per applicant, or per file?
4. Are any PDFs **password-protected** or encrypted? Who provides passwords?
5. Are pages ever **rotated 90°/180°**, upside down, or two pages scanned on one sheet?
6. Any **handwritten** pages or handwritten values on forms (application id, names)? Handwriting is out of scope for this OCR.
7. Scan quality: typical DPI, colour vs b/w, stamps/watermarks over tables?
8. Any non-English pages (regional-language forms, Hindi stamps)?

## Application fields
9. What exactly is the application id format (prefix, length, characters)? Is it printed on every page or only the form?
10. How do **co-applicants** appear: same form section, separate form page, separate file? Can there be more than one co-applicant?
11. Which name is authoritative if the form, the statement account name and KYC documents differ?
12. Full list of label variants used on their forms ("App No", "LAN", "Ref No"...).

## Output & acceptance
13. Expected output schema and format (CSV/Excel/JSON/DB)? One row per file, per applicant, or per account?
14. Do they need every transaction row, or just: statement present (Y/N), months covered, period, bank, account holder, opening/closing balance?
15. Definition of "months covered": calendar months touched, full months only, or period line on the statement?
16. Accuracy bar per field (e.g. ≥95% for has_statements, ≥90% months ±0, ≥90% app id exact)? What may go to manual review instead?
17. Time budget per file (CPU laptop: ~1–3 s per scanned page here → ~1–2 min per 40-page file)?
18. Who builds the ground truth (we need ~10–20 hand-checked files to measure accuracy)?

## Security & governance
19. Approved machine(s) for processing; is the laptop offline / encrypted disk? Can we install Python packages there?
20. Retention and deletion: when must `outputs/` (OCR text, thumbnails, caches) be deleted? Can extracted text be kept for training a classifier?
21. Can redacted or synthetic look-alike pages leave the premises for tuning? (Default assumption: **no**.)
22. Any audit/logging requirement for who ran what on which file?
