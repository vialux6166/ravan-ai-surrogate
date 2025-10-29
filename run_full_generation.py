"""
Run full QuantumML-1K dataset generation (1000 samples)
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

from generate_quantumml_1k import QuantumML1KGenerator

if __name__ == '__main__':
    print("Starting full QuantumML-1K generation (1000 samples)...")
    print("This may take several minutes...")
    print()
    
    # Initialize generator with full design
    generator = QuantumML1KGenerator('./data/benchmark/quantumml_1k_design.json')
    
    # Generate complete dataset
    dataset = generator.generate_dataset(
        output_path='./data/benchmark/quantumml_1k.h5',
        compute_difficulty=True
    )
    
    print("\n" + "=" * 80)
    print("FULL QUANTUML-1K GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Dataset saved to: ./data/benchmark/quantumml_1k.h5")
    print(f"Metadata saved to: ./data/benchmark/quantumml_1k_metadata.json")
    print(f"Total samples: {len(dataset.X)}")
