#!/bin/bash
cd /home/windows/ravan-quantum-ml

# Backup
cp scripts/generate_data.py scripts/generate_data.py.bak

# Use awk to insert the new elif
awk '
/elif simulator_type == .schrodinger.:/ {
    print "    elif simulator_type == '\''enhanced_quantum_circuit'\'':"
    print "        simulator = EnhancedQuantumCircuitSimulator(use_gpu=False)"
}
{print}
' scripts/generate_data.py.bak > scripts/generate_data.py

echo "Fixed generate_data.py"
