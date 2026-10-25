# Open-source document models, Oct 2024 to Oct 2026: fit for the KYC and bank-statement pipeline

*Research note, 25 Oct 2026. Nothing here has been installed or run on our pages. Numbers are the vendors' own claims unless marked otherwise. Anything marked (guess) is our own estimate.*

## 1. Scope and constraints
- **Input:** 40 poor-quality black-and-white scanned PDFs (English), 20–40 pages each, mixing KYC forms and bank statements.
- **What we need:** find the statement pages, the lead ID, the transaction tables and the months of coverage, then write Excel.
- **Where it runs:** on-prem only, with no external APIs. The laptop runs Windows with 16 GB RAM, a CPU and **no GPU**.
- **Licences:** permissive licences are preferred.
- **Note:** the client said "no LLMs". Any VLM below is a local open-weight model and still needs the client's OK.

## 2. Baseline already chosen
- **Ingest:** pypdfium2 and pdfplumber.
- **Preprocessing:** OpenCV grayscale and deskew.
- **OCR:** RapidOCR with PP-OCRv5 mobile (1.76 s/page, amount accuracy 0.986, balance check 0.993).
- **Page classification:** keyword scoring, with TF-IDF/BERT as the planned fallback.
- **Tables:** rules plus the balance check. PP-StructureV3, img2table and Table Transformer are the candidates.
- **Path B:** a local VLM (Granite-Docling, SmolDocling or Qwen2.5-VL).

The list below is what is new or newer on top of that.

## 3. Shortlist
| Name | Released | Type | Licence | Best pipeline step | CPU on 16 GB? | Verdict |
|---|---|---|---|---|---|---|
| **PP-OCRv6 (via RapidOCR ≥3.9)** | Jun 2026 | OCR det+rec | Apache-2.0 | 2 OCR | Yes (7.7M small) | **Strong trial: drop-in upgrade** |
| **Docling + Heron layout + TableFormer** | Heron Sep 2025 (Docling 2024+) | Layout, tables, PDF→JSON | MIT code; Apache-2.0 / CDLA-P models | 3 page class, 6 tables | Yes (~0.6 s/page layout on 4 threads) | **Strong trial** |
| **PaddleOCR-VL 1.5 / 1.6 (0.9B)** | Oct 2025 → Jan / May 2026 | Doc VLM parser | Apache-2.0 | 6 tables (Path B) | Possible, slow (guess) | **Strong Path B candidate** |
| **MinerU 3.x (pipeline / hybrid)** | 2025–Jun 2026 | PDF→MD/JSON system | Custom Apache-based (was AGPL until Apr 2026) | 1–6 | Basic tier on CPU (ONNX) | Maybe: read licence first |
| **LightOnOCR-2-1B / -3 (0.8B, 1B, 4B)** | Jan 2026 / Oct 2026 | End-to-end OCR VLM | Apache-2.0 | 2/6 (Path B) | 0.8–1B maybe, slow | Maybe later |
| **Indra OCR 1.1 (Datalab)** | Oct 2026 | Tiny CPU OCR + layout | Code Apache; weights modified OpenRAIL-M | 2 OCR, 3 layout | Yes (~210 ms/page on 4 cores, claimed) | Trial technically; **licence risk** |
| **Surya OCR 2 (650M)** | May 2026 | OCR + layout + tables | Code Apache; weights OpenRAIL-M (<$5M) | 2, 6 | Via llama.cpp, slow | Skip (licence) |
| **Granite-Docling-258M** | Sep 2025 | Small doc VLM | Apache-2.0 | 6 (Path B) | Yes, slow | Already in baseline |
| **dots.ocr (1.7B, rednote)** | Jul 2025 | Doc VLM | MIT | 6 | GPU practical | Maybe later |
| **MonkeyOCR v1.5** | 2025 | 3-stage parser | Check repo | 6 | GPU practical | Skip for now |
| **DeepSeek-OCR / OCR-2 (3B)** | Oct 2025 / Jan 2026 | Doc VLM | MIT (v1; check v2) | 6 | GPU (vLLM) | Out of scope on this laptop |
| **olmOCR 2 (7B, AllenAI)** | Oct 2025 | Doc VLM | Apache-2.0 | 6 | No (GPU) | Out of scope |

## 4. Notes per item

**PP-OCRv6 in RapidOCR.**
- RapidOCR v3.9.0 (23 Jun 2026) made PP-OCRv6 det/rec *small* the default. It runs on ONNXRuntime/OpenVINO.
- Model sizes: tiny 1.5M, small 7.7M, medium 34.5M parameters.
- Paddle's own numbers: medium scores +5.1% recognition and +4.6% detection over PP-OCRv5_server; small scores 84.1 Hmean and 81.3% recognition.
- It needs **no new architecture**. We bump `rapidocr`, select v6, and re-run our benchmark (amount accuracy, balance check, s/page).
- It is multilingual (50 languages), not English-only, so check it against our English pages.
- Pipeline step: 2.

**Docling (IBM / LF AI) with Heron layout and TableFormer.**
- Heron (RT-DETRv2) became the default layout model in Sep 2025. Vendor numbers: DocLayNet mAP 69.9% vs 50.5% for the old model, and 0.64 s/image on 4 EPYC threads (batch 32).
- TableFormer (fast or accurate) runs on CPU.
- It takes PDFs or images and outputs DoclingDocument JSON, Markdown and tables as dataframes.
- RapidOCR can be plugged in as its OCR engine, so our OCR stays.
- Uses for us:
  - Layout labels (table, page-header, key-value) can feed page classification.
  - TableFormer is a step-6 alternative to PP-StructureV3, without Paddle and on Python 3.14-friendly torch (check this).
- Risk: it was trained on reports and papers more than on noisy bank scans.
- Pipeline steps: 3 and 6.

**PaddleOCR-VL 0.9B (v1 Oct 2025, v1.5 Jan 2026, v1.6 May 2026).**
- A 0.9B parser that is SOTA on OmniDocBench (vendor figures: 94.5% for v1.5 and 96.3% for v1.6 on v1.6 of the benchmark).
- v1.5 added robustness to scanned, skewed and warped pages (the Real5 benchmark), plus seal recognition and text spotting. Seals and skew both matter for us.
- Licence: Apache-2.0.
- CPU inference is possible but slow; budget tens of seconds per page (guess). Use it only on statement pages that fail `balance_ok`.
- It needs the Paddle venv (Python 3.12).
- This is the best new candidate for **Path B**, ahead of Qwen2.5-VL-3B.

**MinerU 3.x (OpenDataLab).**
- A full PDF→Markdown/JSON system with cross-page table merging. That helps with statements that span pages.
- The "Basic" tier runs on CPU with ONNX layout, OCR and SLANet tables. The VLM tier (MinerU2.5-Pro 1.2B) runs via llama.cpp.
- Licence: AGPL up to 2.x. From 3.1 (Apr 2026) it uses a *custom Apache-based licence with extra conditions*, so **legal needs to read it** before client use.
- Pipeline steps: 1–6, as a comparison system.

**LightOnOCR-2-1B (Jan 2026) and LightOnOCR-3 (Oct 2026: 0.8B, 1B, 4B).**
- Apache-2.0 end-to-end OCR. v3 adds labelled layout boxes.
- The vendor claims the 2-1B is state of the art on olmOCR-bench (83.2).
- These are VLMs, so they carry the same client caveat. On CPU only the 0.8B and 1B sizes are realistic, and they are slow.
- Treat them as a Path B alternative to PaddleOCR-VL.

**Indra OCR 1.1 (Datalab, 19 Oct 2026, released this week).**
- A 1M-parameter OCR model (7M for the whole pipeline) on ONNX with no torch. It outputs layout, reading order and JSON boxes.
- Vendor claims:
  - About 210 ms/page on 4 cores, and about 10× PP-OCR text models (their test used RapidOCR PP-OCRv6 small).
  - 66.7 on OmniParseBench, against 58.4 for PP-StructureV3.
- Its own stated limit: complex tables lose structure.
- **Licence:** the weights are modified OpenRAIL-M. That means free only under $20M funding/revenue, and no use that competes with their API.
- SG Analytics/Straive almost certainly exceeds that, so a commercial licence would be needed. It is new and unproven; benchmark-only for now.

**Surya OCR 2 (May 2026).** It is strong (olmOCR-bench 83.3), but it has the same Datalab weight licence (<$5M), and CPU use goes through llama.cpp. Skip.

**dots.ocr, MonkeyOCR, DeepSeek-OCR 1/2, olmOCR 2.** These are good parsers, but they need 1.7–7B parameters and realistically vLLM on a GPU. They are out of scope for this laptop. Revisit if the client provides a GPU box.

## 5. Recommended next trials (in order)
1. **Upgrade RapidOCR → PP-OCRv6 small, then medium.** Same code, re-run our benchmark. This is the cheapest likely win.
2. **Docling with Heron + TableFormer (OCR = RapidOCR)** on 20–30 hand-checked statement pages. Compare it against PP-StructureV3 and our rules on rows found, amount accuracy, `balance_ok`, s/page and peak RAM.
3. **Use Heron layout labels as a feature in page classification** (step 3), alongside the keyword score, before reaching for BERT.
4. **PaddleOCR-VL-1.5/1.6 as Path B** on pages that fail `balance_ok`. Only once the client confirms local VLMs are allowed.
5. **MinerU 3.x Basic (CPU)** as an end-to-end comparison, *after* legal reviews its licence. Cross-page table merging is the reason to try it.
6. Optional: **Indra OCR** speed and accuracy benchmark for information only. Do not adopt it without a commercial licence.

## 6. Out of scope or skip
- **Need a GPU:** olmOCR 2 (7B), DeepSeek-OCR/OCR-2 (3B, vLLM), dots.ocr (1.7B), MonkeyOCR, Qwen3-VL and other large general VLMs, HunyuanOCR, Chandra (9B, and its licence is restricted).
- **Licence-restricted:** Surya 2, Indra and Marker/Chandra weights (Datalab OpenRAIL-M with revenue caps), and PyMuPDF (AGPL).
- **Closed APIs** are excluded by the client: Mistral OCR, Azure/Google/AWS document AI, Gemini and GPT.

## 7. Sources
- RapidOCR v3.9.0 (PP-OCRv6 default): https://github.com/RapidAI/RapidOCR/releases/tag/v3.9.0
- PP-OCRv6: https://huggingface.co/blog/PaddlePaddle/pp-ocrv6 and https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md
- Docling Heron PR: https://github.com/docling-project/docling/pull/1971
- Docling layout report: https://arxiv.org/abs/2509.11720
- Docling model catalog: https://docling-project.github.io/docling/usage/model_catalog/
- Docling model licences: https://github.com/docling-project/docling.rs/blob/master/docs/MODELS_NOTICE.md
- PaddleOCR-VL: https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html
- PaddleOCR-VL-1.5 paper: https://arxiv.org/abs/2601.21957
- MinerU: https://github.com/opendatalab/MinerU
- MinerU changelog: https://opendatalab.github.io/MinerU/reference/changelog/
- MinerU tiers: https://opendatalab.github.io/MinerU/usage/tiers/
- LightOnOCR-2: https://huggingface.co/blog/lightonai/lightonocr-2
- LightOnOCR-3: https://huggingface.co/blog/lightonai/lightonocr-3
- Indra OCR: https://github.com/datalab-to/indra and https://www.datalab.to/blog/indra-ocr-1-1
- Surya 2: https://www.datalab.to/blog/surya-2
- DeepSeek-OCR-2: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 and https://arxiv.org/abs/2601.20552
