"""python run_engine.py ENGINE PREP(raw|prep) -> results/raw/ENGINE__PREP.json"""
import json, os, sys, time, resource, glob
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_THREAD_LIMIT"):
    os.environ[v] = os.environ.get("OCR_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import cv2
from engines import ENGINES
from kyctrial.config import DEFAULTS
from kyctrial.preprocess import preprocess

name, prep = sys.argv[1], sys.argv[2]
levels = sys.argv[3].split(",") if len(sys.argv) > 3 else ["clean", "bw200", "noisy_skew", "faded_blur", "low_contrast"]
t0 = time.time(); run = ENGINES[name](); load_s = time.time() - t0
out = {"engine": name, "prep": prep, "load_s": load_s, "pages": {}}
for lv in levels:
    for f in sorted(glob.glob(f"data/{lv}/*.png")):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if prep == "prep":      # kit default: deskew + median denoise + adaptive threshold
            img, _ = preprocess(img, DEFAULTS)
        elif prep == "deskew":  # deskew only, keep grayscale
            c = {**DEFAULTS, "preprocess": {**DEFAULTS["preprocess"], "denoise": False, "threshold": False}}
            img, _ = preprocess(img, c)
        t = time.time()
        try:
            words = run(img); err = None
        except Exception as e:
            words, err = [], repr(e)
        out["pages"][f"{lv}/{os.path.basename(f)[:-4]}"] = {"words": words, "sec": time.time() - t, "err": err}
        print(name, prep, lv, os.path.basename(f), f"{time.time()-t:.1f}s", len(words), err or "", flush=True)
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
os.makedirs("results/raw", exist_ok=True)
json.dump(out, open(f"results/raw/{name}__{prep}.json", "w"))
print("DONE", name, prep, out["peak_rss_mb"])
