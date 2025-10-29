"""
VRAM Manager for Ravan Quantum-ML System
Manages workload modes and prevents VRAM conflicts

Enhanced with LLM integration (Task 15.3)
"""

import logging
import gc
import torch
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.llm_adapter import LLMAdapter


class WorkloadMode(Enum):
    """Workload modes for VRAM management"""
    SIMULATION = "simulation"  # Quantum simulations + data generation
    TRAINING = "training"      # ML model training
    INFERENCE = "inference"    # ML model inference (lightweight)
    LLM = "llm"               # LLM operations (mutually exclusive)


class VRAMManager:
    """
    Manages VRAM allocation across different workload modes
    
    Ensures mutual exclusion between LLM and simulation/training workloads
    to prevent GPU out-of-memory errors on RTX 3090 (24 GB VRAM).
    """
    
    def __init__(self, gpu_accelerator=None):
        """
        Initialize VRAM manager
        
        Args:
            gpu_accelerator: GPUAccelerator instance
        """
        from gpu_accelerator import GPUAccelerator
        self.gpu = gpu_accelerator or GPUAccelerator()
        self.current_mode: Optional[WorkloadMode] = None
        self.logger = logging.getLogger('ravan.vram')
        
        # LLM adapter instance (lazy loaded)
        self.llm_adapter: Optional['LLMAdapter'] = None
        
        # VRAM requirements for each mode (in GB)
        self.mode_vram_requirements = {
            WorkloadMode.SIMULATION: 8.0,   # Qiskit + CuPy + overhead
            WorkloadMode.TRAINING: 10.0,    # PyTorch + XGBoost + data
            WorkloadMode.INFERENCE: 3.0,    # Lightweight prediction
            WorkloadMode.LLM: 22.0          # Qwen 30B 4-bit quantization
        }
        
        self.logger.info("VRAM Manager initialized")
    
    def request_mode(self, mode: WorkloadMode) -> bool:
        """
        Request to switch to a workload mode
        
        Args:
            mode: Desired workload mode
            
        Returns:
            True if mode switch successful
            
        Raises:
            RuntimeError: If mode switch not allowed due to VRAM constraints
        """
        required_vram = self.mode_vram_requirements[mode]
        
        # Check if switching from LLM to other modes
        if self.current_mode == WorkloadMode.LLM and mode != WorkloadMode.LLM:
            raise RuntimeError(
                "Cannot switch from LLM mode to other modes without unloading LLM. "
                "Call unload_llm() first."
            )
        
        # Check if switching to LLM from other modes
        if mode == WorkloadMode.LLM and self.current_mode is not None:
            free_vram = self.gpu.get_free_vram_gb()
            if free_vram < required_vram:
                raise RuntimeError(
                    f"Cannot load LLM. Required: {required_vram} GB, "
                    f"Available: {free_vram:.1f} GB. "
                    f"Close simulation/training processes first."
                )
        
        # Verify VRAM availability
        self.gpu.check_vram_available(required_vram, mode.value)
        
        # Update current mode
        old_mode = self.current_mode
        self.current_mode = mode
        
        # Mark LLM as loaded if switching to LLM mode
        if mode == WorkloadMode.LLM:
            self.gpu.llm_loaded = True
        
        # Log mode switch
        if old_mode:
            self.logger.info(f"Mode switched: {old_mode.value} → {mode.value}")
        else:
            self.logger.info(f"Mode activated: {mode.value}")
        
        self.gpu.log_vram_usage(f"Mode: {mode.value}")
        return True
    
    def load_llm(self, llm_adapter: 'LLMAdapter') -> bool:
        """
        Load LLM with VRAM management
        
        Args:
            llm_adapter: LLMAdapter instance to load
            
        Returns:
            True if loaded successfully
            
        Raises:
            RuntimeError: If cannot switch to LLM mode
        """
        # Request LLM mode (checks VRAM availability)
        self.request_mode(WorkloadMode.LLM)
        
        # Store adapter reference
        self.llm_adapter = llm_adapter
        
        # Load the model
        self.logger.info("Loading LLM model...")
        success = llm_adapter.load()
        
        if success:
            self.logger.info("✓ LLM loaded successfully")
            self.gpu.log_vram_usage("After LLM load")
        else:
            self.logger.error("✗ LLM loading failed")
            self.current_mode = None
            self.llm_adapter = None
        
        return success
    
    def unload_llm(self):
        """
        Explicitly unload LLM to free VRAM
        
        This method should be called before switching from LLM mode
        to simulation or training modes.
        """
        if self.current_mode == WorkloadMode.LLM:
            self.logger.info("Unloading LLM and freeing VRAM...")
            
            # Unload LLM adapter if present
            if self.llm_adapter is not None:
                self.llm_adapter.unload()
                self.llm_adapter = None
            
            # Trigger garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.gpu.optimize_memory()
            
            # Mark LLM as unloaded
            self.gpu.llm_loaded = False
            self.current_mode = None
            
            self.logger.info("LLM unloaded, VRAM freed")
            self.gpu.log_vram_usage("After LLM unload")
        else:
            self.logger.warning("unload_llm() called but LLM mode not active")
    
    def get_current_mode(self) -> Optional[WorkloadMode]:
        """
        Get the current workload mode
        
        Returns:
            Current workload mode or None
        """
        return self.current_mode
    
    def is_llm_mode(self) -> bool:
        """
        Check if currently in LLM mode
        
        Returns:
            True if in LLM mode
        """
        return self.current_mode == WorkloadMode.LLM
    
    def can_run_simulation(self) -> bool:
        """
        Check if simulation can run in current mode
        
        Returns:
            True if simulation allowed
        """
        return self.current_mode != WorkloadMode.LLM
    
    def can_run_training(self) -> bool:
        """
        Check if training can run in current mode
        
        Returns:
            True if training allowed
        """
        return self.current_mode != WorkloadMode.LLM
    
    def get_vram_budget(self, mode: Optional[WorkloadMode] = None) -> float:
        """
        Get VRAM budget for a mode
        
        Args:
            mode: Workload mode (uses current mode if None)
            
        Returns:
            VRAM budget in GB
        """
        if mode is None:
            mode = self.current_mode
        
        if mode is None:
            return 0.0
        
        return self.mode_vram_requirements.get(mode, 0.0)
    
    def get_llm_adapter(self) -> Optional['LLMAdapter']:
        """
        Get the current LLM adapter instance
        
        Returns:
            LLMAdapter instance if loaded, None otherwise
        """
        return self.llm_adapter
    
    def query_llm(self, prompt: str, **kwargs) -> str:
        """
        Query LLM if loaded
        
        Args:
            prompt: Query prompt
            **kwargs: Additional arguments for query
            
        Returns:
            LLM response
            
        Raises:
            RuntimeError: If LLM not loaded
        """
        if not self.is_llm_mode() or self.llm_adapter is None:
            raise RuntimeError(
                "LLM not loaded. Call load_llm() first."
            )
        
        return self.llm_adapter.query(prompt, **kwargs)
    
    def get_vram_status(self) -> dict:
        """
        Get comprehensive VRAM status
        
        Returns:
            Dictionary with VRAM information
        """
        status = {
            'current_mode': self.current_mode.value if self.current_mode else None,
            'llm_loaded': self.llm_adapter is not None and self.llm_adapter.is_loaded,
        }
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            allocated_vram = torch.cuda.memory_allocated(device) / (1024**3)
            free_vram = total_vram - allocated_vram
            
            status.update({
                'total_vram_gb': total_vram,
                'allocated_vram_gb': allocated_vram,
                'free_vram_gb': free_vram,
                'vram_utilization_pct': (allocated_vram / total_vram) * 100
            })
        
        return status
