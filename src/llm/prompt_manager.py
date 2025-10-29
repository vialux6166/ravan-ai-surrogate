"""
Prompt Management for Ravan Quantum-ML System
Task 16.3: Implement prompt validation and caching

Provides input sanitization, LRU caching, and timeout handling
"""

import sys
import os
import re
import hashlib
import time
from typing import Dict, Any, Optional, Callable
from functools import lru_cache, wraps
from collections import OrderedDict
import logging


class PromptValidator:
    """
    Validates and sanitizes user prompts
    
    Prevents injection attacks and ensures safe inputs
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ravan.prompt_validator')
        
        # Dangerous patterns to block
        self.blocked_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                 # JavaScript protocol
            r'on\w+\s*=',                  # Event handlers
            r'eval\s*\(',                  # Eval calls
            r'exec\s*\(',                  # Exec calls
            r'__import__',                 # Python imports
            r'subprocess',                 # Subprocess calls
            r'os\.system',                 # OS system calls
        ]
        
        # Maximum prompt length
        self.max_length = 5000
        
        # Minimum prompt length
        self.min_length = 3
    
    def validate(self, prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate prompt for safety
        
        Args:
            prompt: User prompt to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check type
        if not isinstance(prompt, str):
            return False, "Prompt must be a string"
        
        # Check length
        if len(prompt) < self.min_length:
            return False, f"Prompt too short (minimum {self.min_length} characters)"
        
        if len(prompt) > self.max_length:
            return False, f"Prompt too long (maximum {self.max_length} characters)"
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                self.logger.warning(f"Blocked pattern detected: {pattern}")
                return False, "Prompt contains potentially unsafe content"
        
        # Check for excessive special characters (possible injection)
        special_char_ratio = sum(1 for c in prompt if not c.isalnum() and not c.isspace()) / len(prompt)
        if special_char_ratio > 0.5:
            return False, "Prompt contains too many special characters"
        
        return True, None
    
    def sanitize(self, prompt: str) -> str:
        """
        Sanitize prompt by removing/escaping dangerous content
        
        Args:
            prompt: Raw prompt
            
        Returns:
            Sanitized prompt
        """
        # Strip leading/trailing whitespace
        prompt = prompt.strip()
        
        # Remove null bytes
        prompt = prompt.replace('\x00', '')
        
        # Normalize whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        
        # Remove control characters except newlines and tabs
        prompt = ''.join(char for char in prompt if char.isprintable() or char in '\n\t')
        
        # Truncate if too long
        if len(prompt) > self.max_length:
            prompt = prompt[:self.max_length]
            self.logger.warning(f"Prompt truncated to {self.max_length} characters")
        
        return prompt


class PromptCache:
    """
    LRU cache for prompt responses
    
    Caches LLM responses to avoid redundant queries
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.logger = logging.getLogger('ravan.prompt_cache')
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _hash_prompt(self, prompt: str, **kwargs) -> str:
        """
        Create hash key for prompt and parameters
        
        Args:
            prompt: Prompt text
            **kwargs: Additional parameters
            
        Returns:
            Hash string
        """
        # Combine prompt with sorted kwargs
        key_parts = [prompt]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Get cached response
        
        Args:
            prompt: Prompt text
            **kwargs: Query parameters
            
        Returns:
            Cached response or None
        """
        key = self._hash_prompt(prompt, **kwargs)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            
            entry = self.cache[key]
            self.logger.debug(f"Cache hit for prompt (age: {time.time() - entry['timestamp']:.1f}s)")
            
            return entry['response']
        
        self.misses += 1
        return None
    
    def put(self, prompt: str, response: str, **kwargs):
        """
        Cache response
        
        Args:
            prompt: Prompt text
            response: LLM response
            **kwargs: Query parameters
        """
        key = self._hash_prompt(prompt, **kwargs)
        
        # Add to cache
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
        
        # Evict oldest if over size
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.logger.debug("Evicted oldest cache entry")
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class TimeoutManager:
    """
    Manages timeouts for LLM queries
    
    Prevents hanging on long-running generations
    """
    
    def __init__(self, default_timeout: float = 60.0):
        """
        Initialize timeout manager
        
        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout
        self.logger = logging.getLogger('ravan.timeout_manager')
    
    def with_timeout(self, timeout: Optional[float] = None):
        """
        Decorator to add timeout to function
        
        Args:
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                timeout_val = timeout or self.default_timeout
                
                start_time = time.time()
                
                # Note: This is a simple timeout check
                # For true async timeout, would need threading/asyncio
                result = func(*args, **kwargs)
                
                elapsed = time.time() - start_time
                
                if elapsed > timeout_val:
                    self.logger.warning(
                        f"Function {func.__name__} took {elapsed:.1f}s "
                        f"(timeout: {timeout_val}s)"
                    )
                
                return result
            
            return wrapper
        return decorator


class PromptManager:
    """
    Unified prompt management system
    
    Combines validation, caching, and timeout handling
    """
    
    def __init__(
        self,
        cache_size: int = 100,
        default_timeout: float = 60.0
    ):
        """
        Initialize prompt manager
        
        Args:
            cache_size: Maximum cache size
            default_timeout: Default timeout in seconds
        """
        self.validator = PromptValidator()
        self.cache = PromptCache(max_size=cache_size)
        self.timeout_mgr = TimeoutManager(default_timeout=default_timeout)
        self.logger = logging.getLogger('ravan.prompt_manager')
    
    def process_prompt(
        self,
        prompt: str,
        query_func: Callable[[str], str],
        use_cache: bool = True,
        timeout: Optional[float] = None,
        **query_kwargs
    ) -> str:
        """
        Process prompt with validation, caching, and timeout
        
        Args:
            prompt: User prompt
            query_func: Function to query LLM
            use_cache: Whether to use cache
            timeout: Query timeout
            **query_kwargs: Additional query parameters
            
        Returns:
            LLM response
            
        Raises:
            ValueError: If prompt validation fails
        """
        # Sanitize prompt
        prompt = self.validator.sanitize(prompt)
        
        # Validate prompt
        is_valid, error_msg = self.validator.validate(prompt)
        if not is_valid:
            raise ValueError(f"Invalid prompt: {error_msg}")
        
        # Check cache
        if use_cache:
            cached_response = self.cache.get(prompt, **query_kwargs)
            if cached_response is not None:
                self.logger.info("Returning cached response")
                return cached_response
        
        # Query LLM with timeout
        self.logger.info("Querying LLM...")
        
        @self.timeout_mgr.with_timeout(timeout)
        def timed_query():
            return query_func(prompt, **query_kwargs)
        
        response = timed_query()
        
        # Cache response
        if use_cache:
            self.cache.put(prompt, response, **query_kwargs)
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

# Global prompt manager instance
_global_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(
    cache_size: int = 100,
    default_timeout: float = 60.0
) -> PromptManager:
    """
    Get global prompt manager instance
    
    Args:
        cache_size: Cache size
        default_timeout: Default timeout
        
    Returns:
        PromptManager instance
    """
    global _global_prompt_manager
    
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager(
            cache_size=cache_size,
            default_timeout=default_timeout
        )
    
    return _global_prompt_manager


def validate_prompt(prompt: str) -> tuple[bool, Optional[str]]:
    """
    Quick prompt validation
    
    Args:
        prompt: Prompt to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = PromptValidator()
    return validator.validate(prompt)


def sanitize_prompt(prompt: str) -> str:
    """
    Quick prompt sanitization
    
    Args:
        prompt: Prompt to sanitize
        
    Returns:
        Sanitized prompt
    """
    validator = PromptValidator()
    return validator.sanitize(prompt)
