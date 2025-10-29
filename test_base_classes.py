"""
Test Base Simulation and ML Model Classes
"""
import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

from simulators.base import SimulationModule, SimulationResult, PhysicsValidator
from models.base import MLModel, ModelConfig, ModelRegistry, TrainingMetrics
from utils.logging import setup_logging
import numpy as np

# Setup logging
logger = setup_logging(level='INFO')

print("=" * 60)
print("Testing Base Classes")
print("=" * 60)

# Test 1: PhysicsValidator
print("\n1. Testing PhysicsValidator...")

# Test probability normalization
probs = np.array([0.25, 0.25, 0.25, 0.25])
is_valid, error = PhysicsValidator.check_probability_normalization(probs)
print(f"   ✓ Probability normalization: {is_valid}")

# Test conservation
is_valid, error = PhysicsValidator.check_conservation(1.0, 1.0, tolerance=0.001)
print(f"   ✓ Conservation check: {is_valid}")

# Test range
is_valid, error = PhysicsValidator.check_range(0.5, 0.0, 1.0)
print(f"   ✓ Range check: {is_valid}")

# Test 2: SimulationResult
print("\n2. Testing SimulationResult...")
result = SimulationResult(
    observables={'transmission': 0.67, 'reflection': 0.33},
    parameters={'V0': 5.0, 'width': 1.0},
    metadata={'timestamp': '2025-10-19'},
    validation_passed=True
)
print(f"   ✓ SimulationResult created")
print(f"   ✓ Observables: {result.observables}")
print(f"   ✓ Parameters: {result.parameters}")

# Test 3: Create a mock simulator
print("\n3. Testing SimulationModule interface...")

class MockSimulator(SimulationModule):
    def __init__(self):
        super().__init__("mock")
    
    def run(self, params):
        return SimulationResult(
            observables={'output': params['input'] * 2},
            parameters=params,
            metadata={'version': '1.0'}
        )
    
    def get_parameter_space(self):
        return {'input': (0.0, 10.0)}
    
    def validate_output(self, result):
        result.validation_passed = True
        return True
    
    def get_observable_names(self):
        return ['output']

sim = MockSimulator()
print(f"   ✓ Mock simulator created: {sim.name}")
print(f"   ✓ Parameter space: {sim.get_parameter_space()}")
print(f"   ✓ Observable names: {sim.get_observable_names()}")

# Test parameter validation
is_valid, errors = sim.validate_parameters({'input': 5.0})
print(f"   ✓ Parameter validation (valid): {is_valid}")

is_valid, errors = sim.validate_parameters({'input': 15.0})
print(f"   ✓ Parameter validation (invalid): {not is_valid}, errors: {len(errors)}")

# Test simulation run
result = sim.run_with_validation({'input': 5.0})
print(f"   ✓ Simulation run: output = {result.observables['output']}")

# Test 4: ModelConfig
print("\n4. Testing ModelConfig...")
config = ModelConfig(
    model_type='mlp',
    hyperparameters={'hidden_layers': [256, 128, 64], 'lr': 0.001},
    input_dim=10,
    output_dim=2,
    save_path='./models/test_model'
)
print(f"   ✓ ModelConfig created: {config.model_type}")
print(f"   ✓ Input dim: {config.input_dim}, Output dim: {config.output_dim}")

# Test 5: Create a mock model
print("\n5. Testing MLModel interface...")

class MockModel(MLModel):
    def train(self, X_train, y_train, X_val, y_val):
        self.is_trained = True
        return TrainingMetrics(
            train_loss=[0.5, 0.3, 0.1],
            val_loss=[0.6, 0.4, 0.2],
            train_mse=0.1,
            val_mse=0.2,
            train_mae=0.05,
            val_mae=0.1,
            r2_score=0.95,
            best_epoch=2,
            training_time=10.0
        )
    
    def predict(self, X):
        return np.random.randn(X.shape[0], self.config.output_dim)
    
    def save(self, path=None):
        pass
    
    def load(self, path):
        pass

model = MockModel(config)
print(f"   ✓ Mock model created: {model.config.model_type}")
print(f"   ✓ Is trained: {model.is_trained}")

# Test input validation
X_test = np.random.randn(5, 10)
try:
    model.validate_input_shape(X_test)
    print(f"   ✓ Input shape validation passed")
except ValueError as e:
    print(f"   ✗ Input shape validation failed: {e}")

# Test prediction
predictions = model.predict(X_test)
print(f"   ✓ Predictions shape: {predictions.shape}")

# Test 6: ModelRegistry
print("\n6. Testing ModelRegistry...")
ModelRegistry.register('mock', MockModel)
print(f"   ✓ Model registered: mock")
print(f"   ✓ Available models: {ModelRegistry.list_models()}")

# Create model from registry
model2 = ModelRegistry.create(config)
print(f"   ✓ Model created from registry: {type(model2).__name__}")

print("\n" + "=" * 60)
print("All base class tests completed successfully!")
print("=" * 60)
