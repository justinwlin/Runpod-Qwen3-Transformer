#!/usr/bin/env python3
"""
Quick test script for Pydantic validation without loading models
"""

import sys
import os
sys.path.append('/app' if os.path.exists('/app') else '.')

# Import just the validation part
from pydantic import BaseModel, Field, field_validator
from typing import Optional

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

    @field_validator('max_new_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        return max(1, min(v, 4096))  # Clamp to 1-4096
    
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

def test_validation():
    """Test various validation scenarios"""
    
    print("=== Testing Pydantic V2 Validation ===\n")
    
    # Test 1: Minimal valid request
    print("1. Testing minimal request...")
    try:
        req = GenerationRequest(prompt="Hello world")
        print("✅ Success - uses defaults")
        print(f"   Temperature: {req.temperature}, Max tokens: {req.max_new_tokens}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Full valid request
    print("\n2. Testing full request...")
    try:
        req = GenerationRequest(
            prompt="Write code",
            temperature=0.8,
            max_new_tokens=500,
            top_p=0.95
        )
        print("✅ Success - all parameters accepted")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Out of range values (should be clamped)
    print("\n3. Testing out-of-range clamping...")
    try:
        req = GenerationRequest(
            prompt="Test",
            temperature=5.0,  # Should clamp to 2.0
            top_p=1.5,        # Should clamp to 1.0
            max_new_tokens=10000  # Should clamp to 4096
        )
        print("✅ Success - values clamped:")
        print(f"   Temperature: {req.temperature} (was 5.0)")
        print(f"   Top_p: {req.top_p} (was 1.5)")
        print(f"   Max tokens: {req.max_new_tokens} (was 10000)")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Missing required field
    print("\n4. Testing missing prompt (should fail)...")
    try:
        req = GenerationRequest(temperature=0.5)
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✅ Correctly failed: {e}")
    
    # Test 5: Invalid types
    print("\n5. Testing invalid types...")
    try:
        req = GenerationRequest(
            prompt="Test",
            temperature="hot"  # String instead of float
        )
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✅ Correctly failed: {e}")
    
    print("\n=== Validation Tests Complete ===")

if __name__ == "__main__":
    test_validation()