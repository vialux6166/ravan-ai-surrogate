"""
LLM Adapter for Ravan Quantum-ML System
Task 15.2: Implement LLMAdapter class

Provides a wrapper for loading and querying Qwen 30B with:
- 4-bit quantization
- VRAM checking
- CPU fallback
- Error handling
"""

import sys
import os
import torch
import gc
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)


@dataclass
class LLMConfig:
    """Configuration for LLM"""
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"  # Updated to Qwen2.5-32B (closest to Qwen3:30B)
    cache_dir: Optional[str] = None
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    device: str = "auto"  # "auto", "cuda", or "cpu"


class LLMAdapter:
    """
    Adapter for Qwen 30B LLM with VRAM management
    
    Features:
    - 4-bit quantization for memory efficiency
    - VRAM checking before loading
    - CPU fallback if insufficient VRAM
    - Query interface with generation config
    - Proper cleanup and unloading
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM adapter
        
        Args:
            config: LLM configuration (uses defaults if None)
        """
        self.config = config or LLMConfig()
        self.logger = logging.getLogger('ravan.llm')
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device = None
        
        # Set cache directory
        if self.config.cache_dir is None:
            self.config.cache_dir = os.path.expanduser("~/.cache/huggingface")
        
        self.logger.info(f"LLMAdapter initialized with model: {self.config.model_name}")
    
    def check_vram_available(self, required_gb: float = 22.0) -> tuple[bool, float]:
        """
        Check if sufficient VRAM is available
        
        Args:
            required_gb: Required VRAM in GB
            
        Returns:
            Tuple of (is_available, free_vram_gb)
        """
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available")
            return False, 0.0
        
        try:
            device = torch.device('cuda:0')
            
            # Clear cache for accurate measurement
            torch.cuda.empty_cache()
            gc.collect()
            
            # Get free VRAM
            total_vram = torch.cuda.get_device_properties(device).total_memory
            allocated_vram = torch.cuda.memory_allocated(device)
            free_vram = (total_vram - allocated_vram) / (1024**3)  # Convert to GB
            
            is_available = free_vram >= required_gb
            
            if is_available:
                self.logger.info(f"✓ Sufficient VRAM: {free_vram:.2f} GB free (need {required_gb:.2f} GB)")
            else:
                self.logger.warning(f"✗ Insufficient VRAM: {free_vram:.2f} GB free (need {required_gb:.2f} GB)")
            
            return is_available, free_vram
            
        except Exception as e:
            self.logger.error(f"Error checking VRAM: {e}")
            return False, 0.0
    
    def load(self, force_cpu: bool = False) -> bool:
        """
        Load LLM model
        
        Args:
            force_cpu: Force CPU loading even if GPU available
            
        Returns:
            True if loaded successfully
        """
        if self.is_loaded:
            self.logger.warning("Model already loaded")
            return True
        
        self.logger.info("Loading LLM model...")
        
        try:
            # Check VRAM if using GPU
            use_gpu = not force_cpu and torch.cuda.is_available()
            
            if use_gpu:
                vram_ok, free_vram = self.check_vram_available(required_gb=22.0)
                
                if not vram_ok:
                    self.logger.warning(
                        f"Insufficient VRAM ({free_vram:.2f} GB). "
                        "Falling back to CPU (will be slow)."
                    )
                    use_gpu = False
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            self.logger.info("✓ Tokenizer loaded")
            
            # Configure quantization if using GPU
            if use_gpu and self.config.load_in_4bit:
                self.logger.info("Configuring 4-bit quantization...")
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                device_map = "auto"
                self.logger.info("✓ 4-bit quantization configured")
            else:
                bnb_config = None
                device_map = None
                self.logger.info("Loading in full precision (CPU mode)")
            
            # Load model
            self.logger.info(f"Loading model: {self.config.model_name}")
            self.logger.info("(This may take 2-5 minutes...)")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set device
            if use_gpu:
                self.device = torch.device('cuda:0')
                vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                self.logger.info(f"✓ Model loaded on GPU (VRAM used: {vram_used:.2f} GB)")
            else:
                self.device = torch.device('cpu')
                self.logger.info("✓ Model loaded on CPU")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.unload()
            return False
    
    def query(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Query the LLM
        
        Args:
            prompt: User prompt/question
            max_new_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            system_prompt: Optional system prompt for context
            
        Returns:
            Generated response text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        
        # Format prompt with system context if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
        
        try:
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            
            # Move to device
            if self.device.type == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(full_prompt):
                response = response[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error during query: {e}")
            raise
    
    def batch_query(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Query LLM with multiple prompts (batch processing)
        
        Args:
            prompts: List of prompts
            max_new_tokens: Maximum tokens per response
            temperature: Sampling temperature
            
        Returns:
            List of responses
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        responses = []
        
        for prompt in prompts:
            response = self.query(prompt, max_new_tokens, temperature)
            responses.append(response)
        
        return responses
    
    def unload(self):
        """Unload model and free VRAM"""
        if not self.is_loaded:
            self.logger.info("Model not loaded, nothing to unload")
            return
        
        self.logger.info("Unloading LLM model...")
        
        # Delete model and tokenizer
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        self.is_loaded = False
        self.device = None
        
        self.logger.info("✓ Model unloaded, VRAM freed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.config.model_name,
            'is_loaded': self.is_loaded,
            'device': str(self.device) if self.device else None,
            'quantization': '4-bit' if self.config.load_in_4bit else 'full',
        }
        
        if self.is_loaded and torch.cuda.is_available():
            info['vram_used_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
            info['vram_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def __enter__(self):
        """Context manager entry"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.unload()
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        if self.is_loaded:
            self.unload()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_llm_adapter(
    model_name: str = "Qwen/Qwen-30B-Chat",
    load_in_4bit: bool = True
) -> LLMAdapter:
    """
    Create and return LLM adapter with default config
    
    Args:
        model_name: HuggingFace model identifier
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        LLMAdapter instance
    """
    config = LLMConfig(
        model_name=model_name,
        load_in_4bit=load_in_4bit
    )
    
    return LLMAdapter(config)


def quick_query(prompt: str, model_name: str = "Qwen/Qwen-30B-Chat") -> str:
    """
    Quick one-off query (loads model, queries, unloads)
    
    Args:
        prompt: Question/prompt
        model_name: Model to use
        
    Returns:
        Response text
    """
    with create_llm_adapter(model_name) as llm:
        return llm.query(prompt)
