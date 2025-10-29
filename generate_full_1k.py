#!/usr/bin/env python3
"""
Generate full QuantumML-1K dataset (1000 samples)
"""
import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

from generate_quantumml_1k import QuantumML1KGenerator

if __name__ == '__main__':
    print("=" * 80)
    print("STARTING FULL QUANTUMML-1K GENERATION (1000 SAMPLES)")
    print("=" * 80)
    print()
    
    generator = QuantumML1KGenerator('./data/benchmark/quantumml_1k_design.json')
    dataset = generator.generate_dataset(
        output_path='./data/benchmark/quantumml_1k.h5',
        compute_difficulty=True
    )
    
    print()
    print("=" * 80)
    print("FULL GENERATION COMPLETE!")
    print("=" * 80)
