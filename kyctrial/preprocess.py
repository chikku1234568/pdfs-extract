"""Step 2: clean up scanned page images before OCR.
Each step is optional (see config['preprocess']). Use --debug to save the
intermediate images and eyeball what each step does to *your* scans."""
import os
import cv2
import numpy as np


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def estimate_skew(gray, max_deg=10):
    """Estimate text skew angle (degrees) using minAreaRect on dark pixels of
    horizontally-smeared text lines. Good enough for small scanner rotations."""
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # smear text into line blobs, then measure the angle of each blob
    smear = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3)))
    contours, _ = cv2.findContours(smear, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        (cx, cy), (w, h), a = cv2.minAreaRect(c)
        if w < h:
            w, h = h, w
            a = a - 90
        if w > 150 and h < 60:  # long thin blobs = text lines
            if a > 45:
                a -= 90
            if a < -45:
                a += 90
            angles.append(a)
    if not angles:
        return 0.0
    ang = float(np.median(angles))
    return ang if abs(ang) <= max_deg else 0.0


def rotate(gray, angle):
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def remove_borders(bw, margin_frac=0.02):
    """Blank out a thin frame around the page (scanner edges / punch holes)."""
    h, w = bw.shape
    my, mx = int(h * margin_frac), int(w * margin_frac)
    out = bw.copy()
    out[:my, :] = 255; out[-my:, :] = 255; out[:, :mx] = 255; out[:, -mx:] = 255
    return out


def preprocess(img, cfg, debug_dir=None, tag="page"):
    p = cfg["preprocess"]
    steps = {}
    gray = to_gray(img); steps["1_gray"] = gray
    angle = 0.0
    if p["deskew"]:
        angle = estimate_skew(gray, p["max_skew_deg"])
        if abs(angle) > 0.1:
            gray = rotate(gray, angle)
        steps["2_deskew"] = gray
    if p["denoise"]:
        gray = cv2.medianBlur(gray, 3)  # kills salt & pepper noise typical of b/w scans
        steps["3_denoise"] = gray
    out = gray
    if p["threshold"]:
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, p["block_size"], p["c"])
        steps["4_threshold"] = out
    if p["remove_borders"]:
        out = remove_borders(out); steps["5_borders"] = out
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for k, v in steps.items():
            cv2.imwrite(os.path.join(debug_dir, f"{tag}_{k}.png"), v)
    return out, angle
