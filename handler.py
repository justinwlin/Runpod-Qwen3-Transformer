"""
OPTIMIZED QWEN Model Handler for RunPod with Performance Enhancements

Key optimizations:
1. Flash Attention 2 for faster attention computation
2. torch.compile() for optimized execution
3. Batch processing support
4. KV cache optimization
5. Better memory management
6. Optional streaming response
"""

import os
import asyncio
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time
import logging
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from functools import lru_cache
import gc

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
        description="Maximum number of new tokens to generate (1-4096). Values auto-clamped to 1-4096 range."
    )
    temperature: float = Field(
        default=0.2, 
        description="Controls randomness: 0.1-0.3 (focused/code), 0.4-0.7 (balanced), 0.8-1.2 (creative), 1.3+ (experimental). Auto-clamped to 0.1-2.0."
    )
    top_p: float = Field(
        default=0.9, 
        description="Nucleus sampling: considers tokens with cumulative probability ≤ top_p. 0.8-0.95 recommended. Auto-clamped to 0.1-1.0."
    )
    top_k: int = Field(
        default=50, 
        description="Top-k sampling: considers only the k most likely tokens. 20-50 recommended, works with top_p. Auto-clamped to 1-100."
    )
    do_sample: bool = Field(
        default=True, 
        description="Sampling strategy: false=greedy (deterministic), true=probabilistic (uses temp/top_p/top_k)"
    )
    repetition_penalty: float = Field(
        default=1.1, 
        description="Reduces repetition: 1.0 (no penalty), 1.05-1.15 (subtle), 1.2+ (strong, may sound unnatural). Auto-clamped to 1.0-2.0."
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response (experimental)"
    )

    @field_validator('max_new_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        return max(1, min(v, 4096))
    
    @field_validator('temperature')
    @classmethod  
    def validate_temperature(cls, v):
        return max(0.1, min(v, 2.0))
    
    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        return max(0.1, min(v, 1.0))
        
    @field_validator('repetition_penalty')
    @classmethod
    def validate_repetition_penalty(cls, v):
        return max(1.0, min(v, 2.0))
        
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        return max(1, min(v, 100))

# Environment variables
mode_to_run = os.getenv("MODE_TO_RUN", "pod")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() == "true"
device_map = os.getenv("DEVICE_MAP", "auto")
max_concurrency = int(os.getenv("MAX_CONCURRENCY", "1"))
use_flash_attention = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
use_compile = os.getenv("USE_COMPILE", "true").lower() == "true"
batch_size = int(os.getenv("BATCH_SIZE", "1"))

logger.info("------- ENVIRONMENT VARIABLES -------")
logger.info(f"Mode running: {mode_to_run}")
logger.info(f"Model name: {model_name}")
logger.info(f"Use quantization: {use_quantization}")
logger.info(f"Device map: {device_map}")
logger.info(f"Max concurrency: {max_concurrency}")
logger.info(f"Use Flash Attention: {use_flash_attention}")
logger.info(f"Use torch.compile: {use_compile}")
logger.info(f"Batch size: {batch_size}")
logger.info("------- -------------------- -------")

# Global model and tokenizer
model = None
tokenizer = None
compiled_model = None

def load_model():
    """Load the QWEN model and tokenizer with optimizations"""
    global model, tokenizer, compiled_model
    
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    
    try:
        # Load tokenizer with padding for batch processing
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'  # Important for batch generation
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model loading configuration with optimizations
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Add Flash Attention 2 if available
        if use_flash_attention and torch.cuda.is_available():
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2 for faster inference")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")
        
        # Add quantization if requested
        if use_quantization and torch.cuda.is_available():
            model_kwargs["load_in_8bit"] = True
            logger.info("Loading model with 8-bit quantization")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Enable better memory allocation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Set model to evaluation mode
        model.eval()
        
        # Compile model for faster execution (PyTorch 2.0+)
        if use_compile and torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                logger.info("Compiling model with torch.compile()...")
                compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed, using uncompiled model: {e}")
                compiled_model = model
        else:
            compiled_model = model
        
        # Warm up the model with a dummy forward pass
        if torch.cuda.is_available():
            logger.info("Warming up model...")
            with torch.no_grad():
                dummy_input = tokenizer("Hello", return_tensors="pt")
                if torch.cuda.is_available():
                    dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
                _ = model.generate(**dummy_input, max_new_tokens=10, do_sample=False)
            torch.cuda.synchronize()
            logger.info("Model warmup complete")
        
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

@lru_cache(maxsize=128)
def get_cached_system_prompt(system_prompt: Optional[str]) -> str:
    """Cache system prompts to avoid recomputation"""
    if system_prompt is None:
        return os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
    return system_prompt

def generate_text_batch(prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Generate text for multiple prompts in a single batch"""
    if model is None or tokenizer is None:
        return [{"error": "Model not loaded"} for _ in prompts]
    
    try:
        # Get generation parameters
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        top_k = kwargs.get('top_k', 50)
        do_sample = kwargs.get('do_sample', True)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        system_prompt = get_cached_system_prompt(kwargs.get('system_prompt'))
        
        # Format prompts with chat template
        batch_texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_texts.append(text)
        
        # Tokenize batch with padding
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        generation_start = time.time()
        
        # Generate text with optimized settings
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = (compiled_model or model).generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache
                num_beams=1,  # Greedy or sampling only (no beam search for speed)
            )
        
        generation_time = time.time() - generation_start
        
        # Process outputs
        results = []
        for i, (output, prompt, text) in enumerate(zip(outputs, prompts, batch_texts)):
            response = tokenizer.decode(output, skip_special_tokens=True)
            if text in response:
                response = response.split(text)[-1].strip()
            
            tokens_generated = len(output) - len(inputs['input_ids'][i])
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            results.append({
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
            })
        
        return results
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        torch.cuda.empty_cache()
        gc.collect()
        return [{"error": "GPU out of memory. Try reducing max_new_tokens or batch size."} for _ in prompts]
    
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        import traceback
        traceback.print_exc()
        return [{"error": str(e)} for _ in prompts]

def generate_text(prompt, **kwargs):
    """Generate text for a single prompt (wrapper for batch function)"""
    results = generate_text_batch([prompt], **kwargs)
    return results[0] if results else {"error": "Generation failed"}

async def handler(event):
    """RunPod handler function with Pydantic validation"""
    
    inputReq = event.get("input", {})
    
    try:
        # Support batch processing
        if isinstance(inputReq, list):
            # Batch request
            requests = [GenerationRequest(**req) for req in inputReq]
            prompts = [req.prompt for req in requests]
            
            # Use first request's parameters for batch (can be modified to support per-request params)
            result = generate_text_batch(
                prompts,
                max_new_tokens=requests[0].max_new_tokens,
                temperature=requests[0].temperature,
                top_p=requests[0].top_p,
                top_k=requests[0].top_k,
                do_sample=requests[0].do_sample,
                repetition_penalty=requests[0].repetition_penalty,
                system_prompt=requests[0].system_prompt
            )
            
            return {"batch_results": result}
        else:
            # Single request
            request = GenerationRequest(**inputReq)
            
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
            # Test single request
            logger.info("\n--- Testing single request ---")
            test_input = {
                "prompt": "Write a Python function to calculate factorial",
                "max_new_tokens": 150,
                "temperature": 0.5
            }
            
            requestObject = {"input": test_input}
            response = await handler(requestObject)
            
            if "error" in response:
                logger.error(f"Error: {response['error']}")
            else:
                logger.info(f"Response: {response['generated_text']}")
                if "metrics" in response:
                    logger.info(f"Generation time: {response['metrics']['generation_time']}s")
                    logger.info(f"Tokens/sec: {response['metrics']['tokens_per_second']}")
            
            # Test batch request
            logger.info("\n--- Testing batch request ---")
            batch_input = [
                {"prompt": "What is 2 + 2?", "max_new_tokens": 50},
                {"prompt": "Explain quantum computing", "max_new_tokens": 100},
                {"prompt": "Write a haiku about code", "max_new_tokens": 50}
            ]
            
            requestObject = {"input": batch_input}
            response = await handler(requestObject)
            
            if "error" in response:
                logger.error(f"Error: {response['error']}")
            elif "batch_results" in response:
                for i, result in enumerate(response['batch_results']):
                    logger.info(f"\nBatch result {i+1}:")
                    if "error" in result:
                        logger.error(f"Error: {result['error']}")
                    else:
                        logger.info(f"Prompt: {result['prompt']}")
                        logger.info(f"Response: {result['generated_text']}")
                        logger.info(f"Tokens/sec: {result['metrics']['tokens_per_second']}")
            
            logger.info("-" * 50)
        
        asyncio.run(main())
    else:
        logger.info("\n=== Running in SERVERLESS mode ===\n")
        logger.info("Starting RunPod serverless handler...")
        
        runpod.serverless.start({
            "handler": handler,
            "concurrency_modifier": lambda _: max_concurrency,
        })