"""
QWEN Model Handler for RunPod

This handler supports ALL QWEN models (Qwen1.5, Qwen2, Qwen2.5, Qwen3) for text generation.

SUPPORTED MODELS:
-----------------
Qwen3 Series (NEWEST - Released 2025):
- Qwen/Qwen3-0.6B              (~1.2GB, fastest)
- Qwen/Qwen3-1.7B              (~3.4GB)
- Qwen/Qwen3-4B                (~8GB)
- Qwen/Qwen3-8B                (~16GB)
- Qwen/Qwen3-14B               (~28GB)
- Qwen/Qwen3-32B               (~64GB)
- Qwen/Qwen3-30B-A3B           (MoE: 30B total, 3B active)
- Qwen/Qwen3-235B-A22B         (MoE: 235B total, 22B active)
- Qwen/Qwen3-4B-Thinking-2507  (Thinking mode variant)

Qwen3-Coder Series (For code generation):
- Qwen/Qwen3-Coder-30B-A3B-Instruct
- Qwen/Qwen3-Coder-480B-A35B-Instruct

Qwen2.5 Series (Stable):
- Qwen/Qwen2.5-0.5B-Instruct  (~1GB, fastest)
- Qwen/Qwen2.5-1.5B-Instruct  (~3GB)
- Qwen/Qwen2.5-3B-Instruct    (~6GB)
- Qwen/Qwen2.5-7B-Instruct    (~15GB)
- Qwen/Qwen2.5-14B-Instruct   (~28GB)
- Qwen/Qwen2.5-32B-Instruct   (~65GB)
- Qwen/Qwen2.5-72B-Instruct   (~145GB)

Qwen2 Series:
- Qwen/Qwen2-0.5B-Instruct
- Qwen/Qwen2-1.5B-Instruct
- Qwen/Qwen2-7B-Instruct
- Qwen/Qwen2-72B-Instruct

Qwen1.5 Series (Legacy):
- Qwen/Qwen1.5-0.5B-Chat
- Qwen/Qwen1.5-1.8B-Chat
- Qwen/Qwen1.5-7B-Chat
- Qwen/Qwen1.5-14B-Chat
- Qwen/Qwen1.5-32B-Chat
- Qwen/Qwen1.5-72B-Chat

INPUT PARAMETERS:
-----------------
Required:
  prompt (str): The input text prompt for generation

Optional:
  system_prompt (str): System message to define model behavior
                      Default: "You are a helpful assistant."
                      
  max_new_tokens (int): Maximum tokens to generate (1-2048)
                        Default: 512
                        
  temperature (float): Controls randomness (0.1-2.0)
                      Lower = more focused, Higher = more creative
                      Default: 0.7
                      
  top_p (float): Nucleus sampling threshold (0.1-1.0)
                Default: 0.9
                
  top_k (int): Top-k sampling parameter (1-100)
              Default: 50
              
  do_sample (bool): Enable sampling (True) or greedy decoding (False)
                   Default: True
                   
  repetition_penalty (float): Penalty for repeating tokens (1.0-2.0)
                             Default: 1.1

ENVIRONMENT VARIABLES:
----------------------
MODEL_NAME: Which model to load (default: Qwen/Qwen2.5-0.5B-Instruct)
MODE_TO_RUN: "pod" for testing, "serverless" for production
USE_QUANTIZATION: "true" to enable 8-bit quantization (reduces memory)
DEVICE_MAP: GPU mapping strategy (default: "auto")
SYSTEM_PROMPT: Default system prompt if not specified in request

EXAMPLE REQUEST:
---------------
{
  "input": {
    "prompt": "Write a Python function to reverse a string",
    "system_prompt": "You are an expert Python programmer.",
    "max_new_tokens": 200,
    "temperature": 0.5,
    "top_p": 0.95,
    "do_sample": true
  }
}

COMPATIBILITY:
--------------
- Works with ANY Hugging Face model that supports AutoModelForCausalLM:
  * All QWEN models (Qwen1.5, Qwen2, Qwen2.5, Qwen3)
  * Llama models (Llama-2, Llama-3, CodeLlama)
  * Mistral models (Mistral-7B, Mixtral)
  * Phi models (Phi-2, Phi-3)
  * DeepSeek models
  * Yi models
  * Gemma models
  * And many more!

- Requirements:
  * transformers>=4.35.0 for Qwen2.5 support
  * transformers>=4.44.0 for Qwen3 support (recommended)
  * GPU recommended for models >3B parameters
  * 8-bit quantization available to reduce memory usage
"""

import os
import asyncio
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import logging
from pydantic import BaseModel, Field, validator
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    """Pydantic model for request validation with reasonable defaults"""
    prompt: str = Field(..., description="The input text prompt for generation", min_length=1)
    system_prompt: Optional[str] = Field(
        default=None, 
        description="System message to define model behavior. Uses environment default if not provided."
    )
    max_new_tokens: int = Field(
        default=1024, 
        description="Maximum number of new tokens to generate (1-4096)", 
        ge=1, 
        le=4096
    )
    temperature: float = Field(
        default=0.2, 
        description="Controls randomness: 0.1-0.3 (focused/code), 0.4-0.7 (balanced), 0.8-1.2 (creative), 1.3+ (experimental)", 
        ge=0.1, 
        le=2.0
    )
    top_p: float = Field(
        default=0.9, 
        description="Nucleus sampling: considers tokens with cumulative probability ≤ top_p. 0.8-0.95 recommended", 
        ge=0.1, 
        le=1.0
    )
    top_k: int = Field(
        default=50, 
        description="Top-k sampling: considers only the k most likely tokens. 20-50 recommended, works with top_p", 
        ge=1, 
        le=100
    )
    do_sample: bool = Field(
        default=True, 
        description="Sampling strategy: false=greedy (deterministic), true=probabilistic (uses temp/top_p/top_k)"
    )
    repetition_penalty: float = Field(
        default=1.1, 
        description="Reduces repetition: 1.0 (no penalty), 1.05-1.15 (subtle), 1.2+ (strong, may sound unnatural)", 
        ge=1.0, 
        le=2.0
    )

    @validator('max_new_tokens')
    def validate_max_tokens(cls, v):
        return min(v, 4096)  # Hard cap at 4096
    
    @validator('temperature', 'top_p', 'repetition_penalty')
    def validate_float_ranges(cls, v, field):
        if field.name == 'temperature':
            return max(0.1, min(v, 2.0))
        elif field.name == 'top_p':
            return max(0.1, min(v, 1.0))
        elif field.name == 'repetition_penalty':
            return max(1.0, min(v, 2.0))
        return v

# Environment variables with support for Qwen1.5, Qwen2, Qwen2.5, and Qwen3
mode_to_run = os.getenv("MODE_TO_RUN", "pod")
# Default to Qwen3-0.6B for testing (smallest Qwen3 model)
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() == "true"
device_map = os.getenv("DEVICE_MAP", "auto")

logger.info("------- ENVIRONMENT VARIABLES -------")
logger.info(f"Mode running: {mode_to_run}")
logger.info(f"Model name: {model_name}")
logger.info(f"Use quantization: {use_quantization}")
logger.info(f"Device map: {device_map}")
logger.info("------- -------------------- -------")

# Global model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the QWEN model and tokenizer"""
    global model, tokenizer
    
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Model loading configuration
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        # Add quantization if requested and GPU is available
        if use_quantization and torch.cuda.is_available():
            model_kwargs["load_in_8bit"] = True
            logger.info("Loading model with 8-bit quantization")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if hasattr(model, 'eval'):
            model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        logger.info(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_text(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, 
                 do_sample=True, top_k=50, repetition_penalty=1.1, 
                 system_prompt=None):
    """Generate text using the loaded model"""
    
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        # Use custom system prompt or default
        if system_prompt is None:
            system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
        
        # Format prompt with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        generation_start = time.time()
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - generation_start
        
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if text in response:
            response = response.split(text)[-1].strip()
        
        # Calculate metrics
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        return {
            "generated_text": response,
            "prompt": prompt,
            "model": model_name,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty
            },
            "metrics": {
                "generation_time": round(generation_time, 2),
                "tokens_generated": tokens_generated,
                "tokens_per_second": round(tokens_per_second, 2)
            }
        }
    
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try reducing max_new_tokens or batch size."}
    
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

async def handler(event):
    """RunPod handler function with Pydantic validation"""
    
    inputReq = event.get("input", {})
    
    try:
        # Validate and parse input using Pydantic
        request = GenerationRequest(**inputReq)
        
        # Generate text with validated parameters
        result = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
            system_prompt=request.system_prompt
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {"error": f"Invalid request parameters: {str(e)}"}

# Load model on startup
if __name__ == "__main__":
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        exit(1)
    
    if mode_to_run == "pod":
        logger.info("\n=== Running in POD mode (testing) ===\n")
        
        async def main():
            # Test prompts
            test_prompts = [
                {
                    "prompt": "Write a Python function to calculate factorial",
                    "max_new_tokens": 150
                },
                {
                    "prompt": "What is 2 + 2?",
                    "max_new_tokens": 50,
                    "temperature": 0.5
                }
            ]
            
            for i, test_input in enumerate(test_prompts, 1):
                logger.info(f"\n--- Test {i}/{len(test_prompts)} ---")
                logger.info(f"Prompt: {test_input['prompt']}")
                
                requestObject = {"input": test_input}
                response = await handler(requestObject)
                
                if "error" in response:
                    logger.error(f"Error: {response['error']}")
                else:
                    logger.info(f"Response: {response['generated_text']}")
                    if "metrics" in response:
                        logger.info(f"Generation time: {response['metrics']['generation_time']}s")
                        logger.info(f"Tokens/sec: {response['metrics']['tokens_per_second']}")
                
                logger.info("-" * 50)
        
        asyncio.run(main())
    else:
        logger.info("\n=== Running in SERVERLESS mode ===\n")
        logger.info("Starting RunPod serverless handler...")
        
        runpod.serverless.start({
            "handler": handler,
            "concurrency_modifier": lambda current: 1,
        })