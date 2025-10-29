#!/usr/bin/env python3
"""
FastAPI Server for Ravan Quantum-ML System
Task 22.3: Create API server for inference

Provides REST API endpoints for:
- Model predictions
- Simulation execution
- Health checks
- Metrics
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
import torch
import time
import logging
from datetime import datetime

# Import Ravan components
import sys
sys.path.insert(0, '.')

from mlp_regressor import MLPRegressor
from xgboost_regressor import XGBoostRegressor
from model_base import ModelConfig
from schrodinger_solver import SchrodingerSolver
from quantum_circuit_simulator import QuantumCircuitSimulator
from harmonic_oscillator import HarmonicOscillator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ravan Quantum-ML API",
    description="REST API for quantum simulation and ML predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global state
models = {}
simulators = {}
request_count = 0
start_time = time.time()


# Pydantic models
class SchrodingerParams(BaseModel):
    V0: float = Field(..., ge=1.0, le=10.0, description="Barrier height")
    barrier_width: float = Field(..., ge=0.5, le=3.0, description="Barrier width")
    k0: float = Field(..., ge=2.0, le=8.0, description="Initial momentum")
    sigma: float = Field(..., ge=0.5, le=2.0, description="Wavepacket width")
    x0: float = Field(..., ge=-10.0, le=0.0, description="Initial position")


class QuantumCircuitParams(BaseModel):
    n_qubits: int = Field(..., ge=2, le=10, description="Number of qubits")
    shots: int = Field(1000, ge=100, le=10000, description="Measurement shots")
    gate_sequence: str = Field("bell", description="Gate sequence type")


class HarmonicOscillatorParams(BaseModel):
    oscillator_length: float = Field(1.0, ge=0.1, le=5.0)
    basis_size: int = Field(10, ge=5, le=50)
    initial_n: int = Field(2, ge=0, le=20)


class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Model name (mlp or xgboost)")
    parameters: List[float] = Field(..., description="Input parameters")


class PredictionResponse(BaseModel):
    predictions: List[float]
    inference_time_ms: float
    model_name: str


class SimulationResponse(BaseModel):
    results: Dict
    simulation_time_ms: float
    simulator_name: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    gpu_available: bool
    models_loaded: List[str]
    simulators_loaded: List[str]
    total_requests: int


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and simulators on startup"""
    global models, simulators
    
    logger.info("Starting Ravan API server...")
    
    # Initialize simulators
    try:
        simulators['schrodinger'] = SchrodingerSolver()
        simulators['quantum_circuit'] = QuantumCircuitSimulator()
        simulators['harmonic_oscillator'] = HarmonicOscillator()
        logger.info(f"Loaded {len(simulators)} simulators")
    except Exception as e:
        logger.error(f"Failed to load simulators: {e}")
    
    # Try to load pre-trained models
    try:
        # MLP model
        mlp_config = ModelConfig(
            model_type='mlp',
            input_dim=5,
            output_dim=4,
            hyperparameters={'hidden_dims': [256, 128, 64]}
        )
        mlp_model = MLPRegressor(mlp_config)
        try:
            # Try to load model from various possible paths
            possible_paths = [
                'models/mlp_schrodinger.pt',
                'models/mlp_final.pt/model.pt',
                'models/mlp/model.pt'
            ]
            loaded = False
            for path in possible_paths:
                try:
                    mlp_model.load(path)
                    models['mlp'] = mlp_model
                    logger.info(f"Loaded MLP model from {path}")
                    loaded = True
                    break
                except:
                    continue
            if not loaded:
                logger.warning("MLP model not found in any expected location")
        except Exception as e:
            logger.warning(f"MLP model loading error: {e}")
        
        # XGBoost model
        xgb_config = ModelConfig(
            model_type='xgboost',
            input_dim=5,
            output_dim=4,
            hyperparameters={'n_estimators': 500},
            save_path='models/xgb_schrodinger.pkl'
        )
        xgb_model = XGBoostRegressor(xgb_config)
        try:
            # Try to load model from various possible paths
            possible_paths = [
                'models/xgb_schrodinger.pkl',
                'models/xgb_final.pkl',
                'models/xgb/model.pkl'
            ]
            loaded = False
            for path in possible_paths:
                try:
                    xgb_model.load(path)
                    models['xgboost'] = xgb_model
                    logger.info(f"Loaded XGBoost model from {path}")
                    loaded = True
                    break
                except:
                    continue
            if not loaded:
                logger.warning("XGBoost model not found in any expected location")
        except Exception as e:
            logger.warning(f"XGBoost model loading error: {e}")
            
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    
    logger.info("Ravan API server started successfully")


# Authentication dependency (simple token-based)
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token"""
    # In production, implement proper token validation
    token = credentials.credentials
    if token != "your-secret-token":  # Replace with proper validation
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global request_count, start_time
    
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        gpu_available=torch.cuda.is_available(),
        models_loaded=list(models.keys()),
        simulators_loaded=list(simulators.keys()),
        total_requests=request_count
    )


# Simulation endpoints
@app.post("/simulate/schrodinger", response_model=SimulationResponse)
async def simulate_schrodinger(params: SchrodingerParams):
    """Run Schrödinger equation simulation"""
    global request_count
    request_count += 1
    
    if 'schrodinger' not in simulators:
        raise HTTPException(status_code=503, detail="Schrödinger simulator not available")
    
    try:
        start = time.time()
        result = simulators['schrodinger'].run(params.dict())
        elapsed = (time.time() - start) * 1000
        
        return SimulationResponse(
            results=result,
            simulation_time_ms=elapsed,
            simulator_name="schrodinger"
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/quantum_circuit", response_model=SimulationResponse)
async def simulate_quantum_circuit(params: QuantumCircuitParams):
    """Run quantum circuit simulation"""
    global request_count
    request_count += 1
    
    if 'quantum_circuit' not in simulators:
        raise HTTPException(status_code=503, detail="Quantum circuit simulator not available")
    
    try:
        start = time.time()
        result = simulators['quantum_circuit'].run(params.dict())
        elapsed = (time.time() - start) * 1000
        
        return SimulationResponse(
            results=result,
            simulation_time_ms=elapsed,
            simulator_name="quantum_circuit"
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/harmonic_oscillator", response_model=SimulationResponse)
async def simulate_harmonic_oscillator(params: HarmonicOscillatorParams):
    """Run harmonic oscillator simulation"""
    global request_count
    request_count += 1
    
    if 'harmonic_oscillator' not in simulators:
        raise HTTPException(status_code=503, detail="Harmonic oscillator simulator not available")
    
    try:
        start = time.time()
        result = simulators['harmonic_oscillator'].run(params.dict())
        elapsed = (time.time() - start) * 1000
        
        # Convert numpy arrays to lists for JSON serialization
        result_serializable = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result_serializable[key] = value.tolist()
            else:
                result_serializable[key] = value
        
        return SimulationResponse(
            results=result_serializable,
            simulation_time_ms=elapsed,
            simulator_name="harmonic_oscillator"
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make ML prediction"""
    global request_count
    request_count += 1
    
    if request.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' not found. Available: {list(models.keys())}"
        )
    
    try:
        # Prepare input
        X = np.array([request.parameters])
        
        # Make prediction
        start = time.time()
        predictions = models[request.model_name].predict(X)
        elapsed = (time.time() - start) * 1000
        
        return PredictionResponse(
            predictions=predictions[0].tolist(),
            inference_time_ms=elapsed,
            model_name=request.model_name
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    global request_count, start_time
    
    metrics = {
        "uptime_seconds": time.time() - start_time,
        "total_requests": request_count,
        "requests_per_second": request_count / (time.time() - start_time),
        "timestamp": datetime.now().isoformat()
    }
    
    if torch.cuda.is_available():
        metrics["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    
    return metrics


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Ravan Quantum-ML API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "simulations": [
                "/simulate/schrodinger",
                "/simulate/quantum_circuit",
                "/simulate/harmonic_oscillator"
            ],
            "predictions": ["/predict"],
            "monitoring": ["/health", "/metrics"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
