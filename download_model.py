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
    
    # Download model
    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✓ Model downloaded")
    
    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model size: {param_count / 1e9:.2f}B parameters")
    
    # Clean up to save memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print("=" * 60)
    print(f"✅ MODEL SUCCESSFULLY PREBAKED: {model_name}")
    print(f"📊 Model size: {param_count / 1e9:.2f}B parameters")
    print("=" * 60)
    
except Exception as e:
    print(f"Error downloading model: {e}")
    print("The container will still build, but model will download on first run")
    sys.exit(0)  # Don't fail the build