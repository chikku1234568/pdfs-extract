"""Step 1: get each page either as words from the PDF text layer (digital pages)
or as an image to OCR (scanned pages).

Coordinates: everything downstream works in *pixel space at the chosen DPI*,
so text-layer words (PDF points) are scaled by dpi/72 to match OCR output.
"""
import pdfplumber
import pypdfium2 as pdfium


def page_count(pdf_path):
    pdf = pdfium.PdfDocument(pdf_path)
    n = len(pdf)
    pdf.close()
    return n


def text_layer_words(pdf_path, page_no, dpi):
    """Return (words, chars) from the embedded text layer. words use the common format."""
    scale = dpi / 72.0
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_no]
        raw = page.extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1.5)
    words = [{
        "text": w["text"],
        "x0": w["x0"] * scale, "y0": w["top"] * scale,
        "x1": w["x1"] * scale, "y1": w["bottom"] * scale,
        "conf": 1.0,
    } for w in raw]
    return words, sum(len(w["text"]) for w in words)


def render_image(pdf_path, page_no, dpi):
    """Render one page to a numpy RGB image."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        img = pdf[page_no].render(scale=dpi / 72.0).to_numpy()  # BGRA/BGRx or RGB
    finally:
        pdf.close()
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def load_page(pdf_path, page_no, cfg):
    """Decide text-layer vs OCR for one page.

    Returns dict(source='text'|'image', words=[...] or None, image=np.ndarray).
    The image is always rendered (needed for the HTML report thumbnails)."""
    dpi = cfg["dpi"]
    image = render_image(pdf_path, page_no, dpi)
    try:
        words, nchars = text_layer_words(pdf_path, page_no, dpi)
    except Exception:  # broken/odd text layer -> fall back to OCR
        words, nchars = [], 0
    if nchars >= cfg["min_text_chars"]:
        return {"source": "text", "words": words, "image": image}
    return {"source": "image", "words": None, "image": image}
