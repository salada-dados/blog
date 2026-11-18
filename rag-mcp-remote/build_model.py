"""Write model/e5-small-int8emb.onnx: e5-small in ONNX, small enough to download on a cold start.

Usage: pip install -r requirements-dev.txt   then   python build_model.py
Then:  python check_model.py   (needs the part 4 database running)

The vocabulary table is 96 of the model's 117.5 million weights. Only that table is stored as int8,
with one scale per row; the transformer stays in fp32. Quantizing the whole model (Hugging Face's
model_qint8_avx512_vnni.onnx, 118 MB) changed the retrieved chunks in 21 of 66 questions, so the
index would have to be rebuilt. This file keeps the same chunks as the index.
"""
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from embedding import HF, fetch

HERE = Path(__file__).parent
OUT = HERE / "model" / "e5-small-int8emb.onnx"
TABLE = "embeddings.word_embeddings.weight"

source = onnx.load(str(fetch(f"{HF}/onnx/model.onnx")))  # fp32, 470 MB
graph = source.graph
table = next(t for t in graph.initializer if t.name == TABLE)
weights = numpy_helper.to_array(table)  # 250,037 tokens x 384

scale = np.abs(weights).max(axis=1, keepdims=True) / 127
quantized = np.round(weights / scale).astype(np.int8)
graph.initializer.remove(table)
graph.initializer.extend([
    numpy_helper.from_array(quantized, TABLE),
    numpy_helper.from_array(scale.astype(np.float32), TABLE + "_scale"),
])

# The Gather that looks up each token now returns int8 rows; turn them back into float
# (row x its scale) before the rest of the network sees them.
gather = next(n for n in graph.node if TABLE in n.input)
token_ids, rows = gather.input[1], gather.output[0]
gather.output[0] = rows + "_int8"
position = list(graph.node).index(gather) + 1
for offset, node in enumerate([
    helper.make_node("Gather", [TABLE + "_scale", token_ids], [rows + "_scale"], axis=0),
    helper.make_node("Cast", [rows + "_int8"], [rows + "_float"], to=TensorProto.FLOAT),
    helper.make_node("Mul", [rows + "_float", rows + "_scale"], [rows]),
]):
    graph.node.insert(position + offset, node)

OUT.parent.mkdir(exist_ok=True)
onnx.save(source, str(OUT))
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(f"{OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB\nsha256 {digest}")
