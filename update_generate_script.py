#!/usr/bin/env python3
with open('/home/windows/ravan-quantum-ml/scripts/generate_data.py', 'r') as f:
    content = f.read()

# Add enhanced simulator to choices
content = content.replace(
    "choices=['quantum_circuit', 'schrodinger', 'harmonic']",
    "choices=['quantum_circuit', 'schrodinger', 'harmonic', 'enhanced_quantum_circuit']"
)

# Add import
content = content.replace(
    'from simulators.harmonic import HarmonicOscillatorModule',
    'from simulators.harmonic import HarmonicOscillatorModule\nfrom simulators.quantum_circuit_enhanced import EnhancedQuantumCircuitSimulator'
)

# Add to simulator initialization
content = content.replace(
    "elif simulator_type == 'harmonic_oscillator':",
    "elif simulator_type == 'enhanced_quantum_circuit':\n        simulator = EnhancedQuantumCircuitSimulator()\n    elif simulator_type == 'harmonic_oscillator':"
)

with open('/home/windows/ravan-quantum-ml/scripts/generate_data.py', 'w') as f:
    f.write(content)

print('Updated generate_data.py')
