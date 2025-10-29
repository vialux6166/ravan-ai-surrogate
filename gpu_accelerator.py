"""
GPU Accelerator for Ravan Quantum-ML System
Manages GPU resources and provides VRAM monitoring
"""

import logging
from typing import Dict, Union, Optional
import torch
import numpy as np


class GPUAccelerator:
    """
    GPU acceleration and VRAM management for RTX 3090
    
    Provides:
    - GPU detection and initialization
    - VRAM monitoring and management
    - Memory optimization utilities
    - Mixed precision training support
    """
    
    def __init__(self, vram_limit_gb: float = 24.0):
        """
        Initialize GPU accelerator
        
        Args:
            vram_limit_gb: Total VRAM limit in GB (default: 24.0 for RTX 3090)
        """
        self.vram_limit_gb = vram_limit_gb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm_loaded = False
        self.logger = logging.getLogger('ravan.gpu')
        
        # Get GPU info on initialization
        self.gpu_info = self._get_gpu_info()
        self.logger.info(f"GPU Accelerator initialized: {self.gpu_info['name']}")
        
        if self.device.type == 'cuda':
            self.logger.info(
                f"VRAM: {self.gpu_info['vram_total_gb']:.2f} GB total, "
                f"{self.get_free_vram_gb():.2f} GB free"
            )
    
    def _get_gpu_info(self) -> Dict:
        """
        Query GPU properties
        
        Returns:
            Dictionary with GPU information
        """
        if torch.cuda.is_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'vram_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'vram_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
                'vram_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3),
                'cuda_version': torch.version.cuda,
                'device_capability': torch.cuda.get_device_capability(0)
            }
        return {
            'name': 'CPU',
            'vram_total_gb': 0,
            'vram_allocated_gb': 0,
            'vram_reserved_gb': 0,
            'cuda_version': None,
            'device_capability': None
        }
    
    def get_free_vram_gb(self) -> float:
        """
        Get available VRAM in GB
        
        Returns:
            Free VRAM in gigabytes
        """
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info(0)
        return free / (1024**3)

    # Backwards-compat helper for tests
    def is_available(self) -> bool:
        return torch.cuda.is_available()
    
    def check_vram_available(self, required_gb: float, operation: str) -> bool:
        """
        Check if sufficient VRAM is available for operation
        
        Args:
            required_gb: Required VRAM in GB
            operation: Description of operation for error message
            
        Returns:
            True if sufficient VRAM available
            
        Raises:
            RuntimeError: If insufficient VRAM
        """
        free_vram = self.get_free_vram_gb()
        if free_vram < required_gb:
            raise RuntimeError(
                f"Insufficient VRAM for {operation}. "
                f"Required: {required_gb:.1f} GB, Available: {free_vram:.1f} GB. "
                f"Close other GPU processes or reduce batch size."
            )
        self.logger.debug(
            f"VRAM check passed for {operation}: "
            f"{required_gb:.1f} GB required, {free_vram:.1f} GB available"
        )
        return True
    
    def prevent_llm_conflict(self):
        """
        Ensure LLM is not loaded when running simulations/training
        
        Raises:
            RuntimeError: If LLM is currently loaded
        """
        if self.llm_loaded:
            raise RuntimeError(
                "LLM is currently loaded and using ~20-22 GB VRAM. "
                "Unload LLM before running simulations or training. "
                "LLM and simulation/training are mutually exclusive."
            )
    
    def to_gpu(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Move data to GPU with VRAM check
        
        Args:
            data: NumPy array or PyTorch tensor
            
        Returns:
            Tensor on GPU device
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device)
    
    def enable_mixed_precision(self) -> torch.cuda.amp.GradScaler:
        """
        Enable FP16 training to reduce VRAM by ~30%
        
        Returns:
            GradScaler for mixed precision training
        """
        if not torch.cuda.is_available():
            self.logger.warning("Mixed precision requested but CUDA not available")
            return None
        
        self.logger.info("Mixed precision (FP16) training enabled")
        return torch.cuda.amp.GradScaler()
    
    def optimize_memory(self):
        """
        Clear cache and optimize VRAM usage
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.debug("GPU cache cleared and synchronized")
    
    def log_vram_usage(self, stage: str):
        """
        Log current VRAM usage for monitoring
        
        Args:
            stage: Description of current stage/operation
        """
        info = self._get_gpu_info()
        free_vram = self.get_free_vram_gb()
        
        self.logger.info(
            f"[{stage}] VRAM: {info['vram_allocated_gb']:.2f} GB allocated, "
            f"{info['vram_reserved_gb']:.2f} GB reserved, "
            f"{free_vram:.2f} GB free"
        )
    
    def get_device(self) -> torch.device:
        """
        Get the current device (cuda or cpu)
        
        Returns:
            PyTorch device
        """
        return self.device
    
    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available
        
        Returns:
            True if CUDA available
        """
        return torch.cuda.is_available()
