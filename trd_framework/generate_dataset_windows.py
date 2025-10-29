#!/usr/bin/env python3
"""
Generate TRD Dataset - Windows Version

Runs directly in Windows to avoid WSL networking issues with Ollama.
"""

import sys
import os

# Add parent directory (project root) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add project root
sys.path.insert(0, current_dir)  # Add trd_framework dir

import json
import time
from pathlib import Path
import argparse
from tqdm import tqdm

# Import from phase1
from phase1_dataset_generation import ResonanceDatasetGenerator
from ravan_trd_integration import RavanTRDIntegration

def main():
    parser = argparse.ArgumentParser(description='Generate TRD Dataset (Windows)')
    parser.add_argument('--target-size', type=int, default=1000, help='Number of examples')
    parser.add_argument('--teacher-model', default='qwen3:30b-a3b', help='Teacher model')
    parser.add_argument('--output-dir', default='trd_framework/large_dataset', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"TRD DATASET GENERATION (Windows)")
    print(f"{'='*70}")
    print(f"Target: {args.target_size} examples")
    print(f"Teacher: {args.teacher_model}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Initialize
    generator = ResonanceDatasetGenerator("http://localhost:11434")
    
    # Try physics mode first, fall back to general mode
    try:
        from ravan_trd_integration import RavanTRDIntegration
        integration = RavanTRDIntegration()
        mode = "physics"
        print("✓ Using physics mode (Ravan simulators)")
    except Exception as e:
        print(f"⚠ Physics mode unavailable: {e}")
        print("✓ Using general mode (multi-domain queries)")
        from general_trd_integration import GeneralTRDIntegration
        integration = GeneralTRDIntegration()
        mode = "general"
    
    # Generate base queries
    print(f"\nStep 1: Generating {mode} queries...")
    if mode == "physics":
        base_queries = integration.create_ravan_specific_dataset(
            output_path=f"{args.output_dir}/base_queries.jsonl",
            n_simulation_queries=60
        )
    else:
        base_queries = integration.create_general_purpose_dataset(
            output_path=f"{args.output_dir}/base_queries.jsonl"
        )
    
    # Calculate variations needed
    n_variations = (args.target_size // len(base_queries)) + 1
    print(f"\nStep 2: Creating {n_variations} variations...")
    
    all_queries = []
    for _ in range(n_variations):
        # Just repeat the base queries for variations
        all_queries.extend(base_queries)
    
    all_queries = all_queries[:args.target_size]
    print(f"✓ Total queries: {len(all_queries)}")
    
    # Generate dataset
    print(f"\nStep 3: Querying teacher model...")
    print(f"Estimated time: {len(all_queries) * 12 / 3600:.1f} hours\n")
    
    output_file = Path(args.output_dir) / "physics_large_dataset.jsonl"
    checkpoint_file = Path(args.output_dir) / "physics_checkpoint.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('processed', 0)
            print(f"✓ Resuming from checkpoint: {start_idx} examples\n")
    
    # Process queries
    processed = start_idx
    queries_text = [q['query'] for q in all_queries[start_idx:]]
    
    with tqdm(total=len(queries_text), desc="Generating") as pbar:
        for i, (query_text, query_data) in enumerate(zip(queries_text, all_queries[start_idx:])):
            try:
                # Query teacher
                response = generator.query_teacher(query_text, args.teacher_model)
                
                if response:
                    # Parse response
                    sections = generator.parse_response(response)
                    
                    # Create example
                    example = {
                        'query': query_text,
                        'initial_thought': sections['initial_thought'].strip(),
                        'chain_of_reasoning': sections['chain_of_reasoning'].strip(),
                        'identified_flaws': sections['identified_flaws'].strip(),
                        'self_correction': sections['self_correction'].strip(),
                        'final_answer': sections['final_answer'].strip(),
                        'full_response': response,
                        'metadata': {
                            'teacher_model': args.teacher_model,
                            'timestamp': time.time(),
                            'query_index': processed + 1
                        },
                        **{k: v for k, v in query_data.items() if k != 'query'}
                    }
                    
                    # Save
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                    
                    processed += 1
                    pbar.update(1)
                    
                    # Checkpoint
                    if processed % 100 == 0:
                        with open(checkpoint_file, 'w') as f:
                            json.dump({'processed': processed, 'timestamp': time.time()}, f)
                        print(f"\n✓ Checkpoint: {processed}/{args.target_size}")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n✗ Error: {e}")
                continue
    
    # Final checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': processed, 'timestamp': time.time(), 'complete': True}, f)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated: {processed} examples")
    print(f"File: {output_file}")
    print(f"Size: {output_file.stat().st_size / (1024**2):.2f} MB")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
