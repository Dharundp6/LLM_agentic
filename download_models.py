"""
Download required models from HuggingFace for Kaggle upload.
Only fetches PyTorch + tokenizer files (skips ONNX/OpenVINO/TF variants).
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

# Allow only the files we actually need
ALLOW = [
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer*",
    "vocab*",
    "spiece*",
    "sentencepiece*",
    "config*",
    "pytorch_model.bin",
    "model.safetensors",
    "source.spm",
    "target.spm",
]

IGNORE = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.msgpack",
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
    "*coreml*",
]

MODELS = {
    "intfloat/multilingual-e5-large":              "models/multilingual-e5-large",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1":  "models/mmarco-reranker",
    "Helsinki-NLP/opus-mt-en-de":                  "models/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-tc-big-en-de":           "models/opus-mt-tc-big-en-de",
    "BAAI/bge-m3":                                 "models/bge-m3",
}

for hf_id, local_dir in MODELS.items():
    path = Path(local_dir)
    if path.exists() and any(path.glob("*.safetensors")) or any(path.glob("pytorch_model.bin")):
        print(f"Already exists: {local_dir}", flush=True)
        continue
    print(f"Downloading {hf_id} -> {local_dir} ...", flush=True)
    try:
        snapshot_download(
            hf_id,
            local_dir=local_dir,
            allow_patterns=ALLOW,
            ignore_patterns=IGNORE,
            max_workers=4,
        )
        print(f"  Done: {local_dir}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        sys.exit(1)

print("\nAll models downloaded.", flush=True)
