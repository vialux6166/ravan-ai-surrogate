"""
Complete QuantumML-1K Generation Script
Step 1: Create benchmark design
Step 2: Generate full 1000-sample dataset
"""

import sys
import os
import warnings

# Suppress NumPy warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def step1_create_design():
    """Step 1: Create benchmark design"""
    logger.info("=" * 80)
    logger.info("STEP 1: Creating QuantumML-1K Benchmark Design")
    logger.info("=" * 80)
    
    from benchmark_dataset_generator import create_quantumml_1k_design
    
    design = create_quantumml_1k_design()
    
    logger.info("✓ Benchmark design created successfully")
    return design


def step2_generate_dataset():
    """Step 2: Generate full dataset"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Generating QuantumML-1K Dataset (1000 samples)")
    logger.info("=" * 80)
    
    from generate_quantumml_1k import QuantumML1KGenerator
    
    # Initialize generator
    generator = QuantumML1KGenerator('./data/benchmark/quantumml_1k_design.json')
    
    # Generate complete dataset
    dataset = generator.generate_dataset(
        output_path='./data/benchmark/quantumml_1k.h5',
        compute_difficulty=True
    )
    
    logger.info("✓ Dataset generation complete")
    return dataset


def main():
    """Main execution function"""
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("QUANTUMML-1K FULL GENERATION")
    print("=" * 80)
    print("This will:")
    print("  1. Create benchmark design (1000 parameter configurations)")
    print("  2. Run simulations for all configurations")
    print("  3. Compute difficulty scores")
    print("  4. Save complete dataset with metadata")
    print("\nEstimated time: 15-30 minutes")
    print("=" * 80)
    print()
    
    try:
        # Step 1: Create design
        design = step1_create_design()
        
        # Step 2: Generate dataset
        dataset = step2_generate_dataset()
        
        # Summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("✓ QUANTUMML-1K GENERATION COMPLETE!")
        print("=" * 80)
        print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        print(f"\nDataset details:")
        print(f"  Samples: {len(dataset.X)}")
        print(f"  Parameters: {len(dataset.parameter_names)}")
        print(f"  Observables: {len(dataset.observable_names)}")
        print(f"\nFiles created:")
        print(f"  - ./data/benchmark/quantumml_1k_design.json")
        print(f"  - ./data/benchmark/quantumml_1k.h5")
        print(f"  - ./data/benchmark/quantumml_1k_metadata.json")
        print("=" * 80)
        
        return dataset
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    dataset = main()
    
    if dataset is not None:
        print("\n✓ Success! QuantumML-1K benchmark is ready for use.")
    else:
        print("\n✗ Generation failed. Check logs for details.")
