"""multilingual-e5-small without PyTorch: ONNX Runtime + the Hugging Face tokenizer.

PyTorch alone is larger than a Vercel Python function may be (500 MB), so the server runs the
model with ONNX Runtime. The model file is the one build_model.py writes: the original fp32
weights, except the vocabulary table, stored as int8 with one scale per row (183 MB, from 470 MB).
check_model.py shows it brings the same 3 chunks as the index for all 66 test questions.

The files are not in the function bundle. They are downloaded on the first search into CACHE_DIR
(/tmp on Vercel, the only writable folder) and stay there while the instance lives.
"""
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HF = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/614241f622f53c4eeff9890bdc4f31cfecc418b3"
TOKENIZER = (f"{HF}/onnx/tokenizer.json", None)
MODEL = (
    os.environ.get("MODEL_URL", ""),
    os.environ.get("MODEL_SHA256") or None,
)
CACHE = Path(os.environ.get("CACHE_DIR", Path(tempfile.gettempdir()) / "rpm-mcp"))


def fetch(url, sha256=None, local=None):
    """Return a local path for url: an existing local file, the cache, or a fresh download."""
    if local and Path(local).exists():
        return Path(local)
    path = CACHE / url.rsplit("/", 1)[-1]
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        part = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as out:
                while block := response.read(1 << 20):
                    out.write(block)
        except BaseException:
            part.unlink(missing_ok=True)  # a failed download must not keep filling /tmp
            raise
        if sha256 and hashlib.sha256(part.read_bytes()).hexdigest() != sha256:
            part.unlink()
            raise RuntimeError(f"{url}: checksum does not match")
        part.replace(path)  # a half-written file never takes the final name
        print(f"downloaded {path.name}: {path.stat().st_size / 1e6:.0f} MB "
              f"in {time.perf_counter() - start:.1f} s; {shutil.disk_usage(CACHE).free / 1e6:.0f} MB "
              f"left in {CACHE}", file=sys.stderr)
    return path


class Embedder:
    def __init__(self, model_path, tokenizer_path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        start = time.perf_counter()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(512)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        print(f"model loaded in {time.perf_counter() - start:.1f} s", file=sys.stderr)

    def encode(self, text):
        """Same vector as SentenceTransformer(...).encode(text, normalize_embeddings=True):
        mean of the token vectors, ignoring padding, then unit length."""
        tokens = self.tokenizer.encode(text)
        ids = np.array([tokens.ids], dtype=np.int64)
        mask = np.array([tokens.attention_mask], dtype=np.int64)
        hidden = self.session.run(None, {
            "input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids),
        })[0]
        vector = (hidden * mask[..., None]).sum(1)[0] / mask.sum()
        return (vector / np.linalg.norm(vector)).astype(np.float32)


_lock = threading.Lock()
_embedder = None


def embedder():
    """Load once per instance. The lock matters: Vercel sends concurrent requests to the same
    instance, and two first searches must not download the model twice."""
    global _embedder
    with _lock:
        if _embedder is None:
            model_url, model_sha = MODEL
            local = os.environ.get("MODEL_PATH") or Path(__file__).parent / "model" / "e5-small-int8emb.onnx"
            if not Path(local).exists() and not model_url:
                raise RuntimeError("set MODEL_PATH or MODEL_URL (run build_model.py first)")
            _embedder = Embedder(fetch(model_url, model_sha, local), fetch(*TOKENIZER))
        return _embedder
