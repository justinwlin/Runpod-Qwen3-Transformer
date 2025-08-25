# QWEN Model RunPod Handler with Prebaked Docker

## Quick Start

### Build Docker Image with Prebaked Model
depot bake all --push

```bash
# Build specific model variants (each creates a separate image)
depot build --platform linux/amd64 -t justinrunpod/qwen:0.6b --build-arg MODEL_NAME=Qwen/Qwen3-0.6B . --push
depot build --platform linux/amd64 -t justinrunpod/qwen:4b --build-arg MODEL_NAME=Qwen/Qwen3-4B . --push  
depot build --platform linux/amd64 -t justinrunpod/qwen:8b --build-arg MODEL_NAME=Qwen/Qwen3-8B . --push
    
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

## API Format & Examples

### Basic Request (minimal)
```json
{
  "input": {
    "prompt": "Write a Python function to reverse a string"
  }
}
```

### Full Request (all parameters)
```json
{
  "input": {
    "prompt": "Explain quantum computing in simple terms",
    "system_prompt": "You are a physics professor explaining to beginners",
    "max_new_tokens": 300,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "do_sample": true,
    "repetition_penalty": 1.2
  }
}
```

### Creative Writing
```json
{
  "input": {
    "prompt": "Write a short story about a robot discovering emotions",
    "system_prompt": "You are a creative fiction writer",
    "max_new_tokens": 500,
    "temperature": 1.0,
    "top_p": 0.9
  }
}
```

### Code Generation (precise)
```json
{
  "input": {
    "prompt": "Create a REST API endpoint in FastAPI for user authentication",
    "system_prompt": "You are an expert Python developer",
    "max_new_tokens": 400,
    "temperature": 0.3,
    "top_p": 0.8,
    "do_sample": false
  }
}
```

### Response Format
```json
{
  "generated_text": "def reverse_string(s):\n    return s[::-1]",
  "prompt": "Write a Python function to reverse a string",
  "model": "Qwen/Qwen3-4B",
  "parameters": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": true,
    "repetition_penalty": 1.1
  },
  "metrics": {
    "generation_time": 1.23,
    "tokens_generated": 15,
    "tokens_per_second": 12.2
  }
}
```

## Parameter Guide

### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | **required** | - | The input text to generate from |
| `system_prompt` | string | `null` | - | System message defining assistant behavior |
| `max_new_tokens` | int | `1024` | 1-4096 | Maximum number of new tokens to generate |

### Sampling Parameters

#### `temperature` (float, default: `0.2`, range: `0.1-2.0`)
Controls randomness in generation:
- **0.1-0.3**: Very focused, deterministic output (good for factual Q&A, code)
- **0.4-0.7**: Balanced creativity and coherence (general use)
- **0.8-1.2**: More creative and diverse (storytelling, brainstorming)
- **1.3-2.0**: Highly creative, potentially incoherent (experimental)

#### `do_sample` (boolean, default: `true`)
Fundamental sampling strategy:
- **`false`**: Greedy decoding - always picks most likely token (deterministic)
- **`true`**: Enables probabilistic sampling (uses temperature, top_p, top_k)

#### `top_p` (float, default: `0.9`, range: `0.1-1.0`)
**Nucleus sampling** - considers tokens whose cumulative probability ≤ top_p:
- **0.1-0.3**: Very conservative, only most likely tokens
- **0.5-0.7**: Moderate diversity
- **0.8-0.95**: Good balance (recommended range)
- **0.96-1.0**: Allows more diverse/risky choices

#### `top_k` (int, default: `50`, range: `1-100`)
**Top-k sampling** - considers only the k most likely tokens:
- **1-10**: Very conservative
- **20-50**: Good balance (recommended)
- **60-100**: More diverse choices
- **Note**: Works together with top_p (intersection of both filters)

#### `repetition_penalty` (float, default: `1.1`, range: `1.0-2.0`)
Reduces repetitive text:
- **1.0**: No penalty (may repeat)
- **1.05-1.15**: Subtle reduction (recommended)
- **1.2-1.5**: Strong penalty (may sound unnatural)
- **1.6-2.0**: Very strong penalty (often breaks coherence)

### Parameter Combinations

#### For **Code Generation** (precise, deterministic):
```json
{
  "temperature": 0.1,
  "top_p": 0.8,
  "top_k": 20,
  "do_sample": false,
  "repetition_penalty": 1.05
}
```

#### For **Creative Writing** (diverse, flowing):
```json
{
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 60,
  "do_sample": true,
  "repetition_penalty": 1.1
}
```

#### For **Factual Q&A** (focused, accurate):
```json
{
  "temperature": 0.3,
  "top_p": 0.85,
  "top_k": 40,
  "do_sample": true,
  "repetition_penalty": 1.05
}
```

#### For **Brainstorming** (highly creative):
```json
{
  "temperature": 1.0,
  "top_p": 0.9,
  "top_k": 80,
  "do_sample": true,
  "repetition_penalty": 1.15
}
```

## Environment Variables

- `MODEL_NAME` - Which model to use (default: Qwen/Qwen3-0.6B)
- `MODE_TO_RUN` - "pod" or "serverless" (default: pod)
- `USE_QUANTIZATION` - "true" to reduce memory usage
- `SYSTEM_PROMPT` - Default system prompt
- `MAX_CONCURRENCY` - Maximum concurrent requests in serverless mode (default: 1)

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