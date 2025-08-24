# QWEN Model RunPod Handler with Prebaked Docker

## Quick Start

### Build Docker Image with Prebaked Model

```bash
# Build with default Qwen3-0.6B (smallest, ~1.2GB)
depot build --platform linux/amd64 \
    -t justinrunpod/qwen:0.6b --build-arg MODEL_SIZE=0.6B \
    -t justinrunpod/qwen:4b --build-arg MODEL_SIZE=4B \
    -t justinrunpod/qwen:8b --build-arg MODEL_SIZE=8B \
    -t justinrunpod/qwen:latest --build-arg MODEL_SIZE=4B \
    . --push
    
## Files

- **handler.py** - Main handler supporting all QWEN models (Qwen1.5, Qwen2, Qwen2.5, Qwen3)
- **Dockerfile** - Builds RunPod-compatible image with prebaked model
- **download_model.py** - Downloads model during Docker build
- **start.sh** - Startup script
- **requirements.txt** - Python dependencies

## Supported Models

### Qwen3 (Latest 2025)
- `Qwen/Qwen3-0.6B` - Smallest, ~1.2GB
- `Qwen/Qwen3-4B` - Balanced, ~8GB
- `Qwen/Qwen3-8B` - Large, ~16GB
- `Qwen/Qwen3-Coder-30B-A3B-Instruct` - For code generation

### Qwen2.5 (Stable)
- `Qwen/Qwen2.5-0.5B-Instruct` - ~1GB
- `Qwen/Qwen2.5-7B-Instruct` - ~15GB

## API Format

```json
{
  "input": {
    "prompt": "Your question here",
    "system_prompt": "You are a helpful assistant",
    "max_new_tokens": 512,
    "temperature": 0.7
  }
}
```

## Environment Variables

- `MODEL_NAME` - Which model to use (default: Qwen/Qwen3-0.6B)
- `MODE_TO_RUN` - "pod" or "serverless" (default: pod)
- `USE_QUANTIZATION` - "true" to reduce memory usage
- `SYSTEM_PROMPT` - Default system prompt

## Building for Different Models

```bash
# Small model for testing
docker build \
  --build-arg MODEL_NAME="Qwen/Qwen3-0.6B" \
  -t qwen-runpod:small .

# Production model
docker build \
  --build-arg MODEL_NAME="Qwen/Qwen3-4B" \
  -t qwen-runpod:prod .

# Coder model with quantization
docker build \
  --build-arg MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  -t qwen-runpod:coder .
```

Then run with quantization:
```bash
docker run --gpus all \
  -e USE_QUANTIZATION="true" \
  qwen-runpod:coder
```

## Deploy to RunPod

1. Build and push to Docker Hub:
```bash
docker build -t yourusername/qwen-runpod:latest .
docker push yourusername/qwen-runpod:latest
```

2. Create RunPod serverless endpoint with your image

3. Set environment variables in RunPod:
   - `MODE_TO_RUN=serverless`
   - `MODEL_NAME=Qwen/Qwen3-4B` (if different from prebaked)

## Notes

- Model is downloaded during Docker build (prebaked)
- First run is instant since model is already in the image
- Image size depends on model (1GB-30GB+)
- Supports ALL Hugging Face models, not just QWEN