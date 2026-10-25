# Getting BERT without Hugging Face (tested 25 Oct 2026)

| Source | How | Tested from our box |
|---|---|---|
| **ModelScope** (Alibaba model hub, Apache-2.0 client) | `pip install modelscope` then `modelscope download --model AI-ModelScope/distilbert-base-uncased --include "*.safetensors" "*.json" "vocab.txt" --local_dir models/distilbert-base-uncased` | **Works.** About 256 MB. Loads with `transformers` using `local_files_only=True`. |
| PaddleNLP weights on Baidu BOS | `https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams` | URL reachable. Needs the PaddleNLP stack. Not wired in. |
| Google original BERT zips (`storage.googleapis.com/bert_models/...`) | TF checkpoints | **403 Forbidden.** Not usable. |

Install on Windows (in the venv used for the notebook): `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install transformers modelscope`.

**Caveats**
- The client network must allow `modelscope.cn`. Ask IT, just as you would for Hugging Face.
- Base DistilBERT is **not trained** to recognise page types. In `main.ipynb` (step 5) it compares each page with a few example texts per class. On the dummy PDF it labelled every page `applicant`, so it is wrong as a standalone classifier.
- For BERT to be useful, fine-tune it on about 50–200 labelled client pages per class. Until then, keywords stay the main method, and the notebook uses BERT only when keywords return `not_needed`.
