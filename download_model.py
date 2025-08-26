#!/usr/bin/env python3
"""
Download and cache QWEN model during Docker build
This "prebakes" the model into the container image
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Get model name from environment
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
cache_dir = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", "/app/cache"))

print("=" * 60)
print(f"🚀 DOWNLOADING MODEL: {model_name}")
print(f"📁 Cache directory: {cache_dir}")
print("=" * 60)

try:
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    print("✓ Tokenizer downloaded")
    
    # Download model weights without loading into memory
    print("Downloading model weights...")
    from huggingface_hub import snapshot_download
    
    # Just download the files without loading the model
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"]  # Skip unnecessary formats
    )
    print("✓ Model files downloaded")
    
    # Clean up
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("=" * 60)
    print(f"✅ MODEL SUCCESSFULLY PREBAKED: {model_name}")
    print("=" * 60)
    
except Exception as e:
    print(f"Error downloading model: {e}")
    print("The container will still build, but model will download on first run")
    sys.exit(0)  # Don't fail the build