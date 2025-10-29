#!/usr/bin/env python3
"""
Generate Large-Scale TRD Dataset

This script generates 100,000+ training examples using the Qwen 30B teacher model.
It uses parallel processing and batching for efficiency.
"""

import sys
sys.path.insert(0, '.')

import json
import time
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from trd_framework.phase1_dataset_generation import ResonanceDatasetGenerator
from trd_framework.ravan_trd_integration import RavanTRDIntegration
from trd_framework.general_trd_integration import GeneralTRDIntegration


class LargeScaleDatasetGenerator:
    """
    Generate large-scale TRD datasets (100k+ examples)
    
    Features:
    - Parallel query generation
    - Batch processing
    - Progress tracking
    - Checkpoint saving
    - Resume capability
    """
    
    def __init__(
        self,
        teacher_url: str = "http://localhost:11434",
        teacher_model: str = "qwen3:30b-a3b",
        mode: str = "physics"
    ):
        # Fix WSL to Windows connection
        if teacher_url == "http://localhost:11434":
            # Try to get Windows host IP from WSL
            try:
                import subprocess
                result = subprocess.run(['cat', '/etc/resolv.conf'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'nameserver' in line:
                        windows_ip = line.split()[1]
                        teacher_url = f"http://{windows_ip}:11434"
                        print(f"✓ Using Windows host IP: {windows_ip}")
                        break
            except:
                pass
        
        self.teacher_url = teacher_url
        self.teacher_model = teacher_model
        self.mode = mode
        self.dataset_generator = ResonanceDatasetGenerator(teacher_url)
        
        if mode == "physics":
            self.integration = RavanTRDIntegration()
        elif mode == "general":
            self.integration = GeneralTRDIntegration()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def generate_query_variations(self, base_queries: List[Dict], n_variations: int = 10) -> List[Dict]:
        """
        Generate variations of base queries to create more training data
        
        For physics mode: Vary simulation parameters
        For general mode: Rephrase questions
        
        Args:
            base_queries: Original queries
            n_variations: Number of variations per query
            
        Returns:
            Expanded query list
        """
        print(f"\nGenerating {n_variations} variations per query...")
        
        all_queries = []
        
        if self.mode == "physics":
            # For physics: Generate more simulation parameter combinations
            for _ in range(n_variations):
                queries = self.integration.generate_physics_grounded_queries(n_samples=60)
                all_queries.extend(queries)
        else:
            # For general: Use base queries multiple times with slight variations
            for _ in range(n_variations):
                all_queries.extend(base_queries)
        
        print(f"✓ Generated {len(all_queries)} total queries")
        return all_queries
    
    def generate_batch(
        self,
        queries: List[Dict],
        batch_size: int = 100,
        output_dir: str = "trd_framework/large_dataset",
        checkpoint_interval: int = 1000
    ):
        """
        Generate dataset in batches with checkpointing
        
        Args:
            queries: List of queries to process
            batch_size: Queries per batch
            output_dir: Output directory
            checkpoint_interval: Save checkpoint every N examples
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_file = Path(output_dir) / f"{self.mode}_large_dataset.jsonl"
        checkpoint_file = Path(output_dir) / f"{self.mode}_checkpoint.json"
        
        # Load checkpoint if exists
        start_idx = 0
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get('processed', 0)
                print(f"\n✓ Resuming from checkpoint: {start_idx} examples processed")
        
        total_queries = len(queries)
        queries_text = [q['query'] for q in queries[start_idx:]]
        
        print(f"\n{'='*70}")
        print(f"LARGE-SCALE DATASET GENERATION")
        print(f"{'='*70}")
        print(f"Mode: {self.mode}")
        print(f"Teacher Model: {self.teacher_model}")
        print(f"Total Queries: {total_queries}")
        print(f"Starting from: {start_idx}")
        print(f"Remaining: {len(queries_text)}")
        print(f"Batch Size: {batch_size}")
        print(f"Output: {output_file}")
        print(f"{'='*70}\n")
        
        # Process in batches
        processed = start_idx
        
        with tqdm(total=len(queries_text), desc="Generating dataset") as pbar:
            for i in range(0, len(queries_text), batch_size):
                batch = queries_text[i:i+batch_size]
                batch_queries = queries[start_idx + i:start_idx + i + len(batch)]
                
                # Generate responses for batch
                for j, (query_text, query_data) in enumerate(zip(batch, batch_queries)):
                    try:
                        # Query teacher
                        response = self.dataset_generator.query_teacher(
                            query_text,
                            self.teacher_model
                        )
                        
                        if response:
                            # Parse response
                            sections = self.dataset_generator.parse_response(response)
                            
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
                                    'teacher_model': self.teacher_model,
                                    'timestamp': time.time(),
                                    'query_index': processed + 1
                                },
                                **{k: v for k, v in query_data.items() if k != 'query'}
                            }
                            
                            # Save incrementally
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                            
                            processed += 1
                            pbar.update(1)
                            
                            # Save checkpoint
                            if processed % checkpoint_interval == 0:
                                with open(checkpoint_file, 'w') as f:
                                    json.dump({
                                        'processed': processed,
                                        'timestamp': time.time(),
                                        'total': total_queries
                                    }, f)
                                print(f"\n✓ Checkpoint saved: {processed}/{total_queries} examples")
                        
                        # Rate limiting
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"\n✗ Error processing query {processed + 1}: {e}")
                        continue
        
        # Final checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed': processed,
                'timestamp': time.time(),
                'total': total_queries,
                'complete': True
            }, f)
        
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Total examples: {processed}")
        print(f"Output file: {output_file}")
        print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")
        print(f"{'='*70}\n")
        
        return output_file
    
    def generate_large_dataset(
        self,
        target_size: int = 100000,
        output_dir: str = "trd_framework/large_dataset"
    ):
        """
        Generate large-scale dataset
        
        Args:
            target_size: Target number of examples
            output_dir: Output directory
        """
        print(f"\n{'='*70}")
        print(f"GENERATING LARGE-SCALE TRD DATASET")
        print(f"{'='*70}")
        print(f"Target size: {target_size:,} examples")
        print(f"Mode: {self.mode}")
        print(f"Teacher: {self.teacher_model}")
        print(f"{'='*70}\n")
        
        # Step 1: Generate base queries
        print("Step 1: Generating base queries...")
        if self.mode == "physics":
            base_queries = self.integration.create_ravan_specific_dataset(
                output_path=f"{output_dir}/base_queries.jsonl",
                n_simulation_queries=60
            )
        else:
            base_queries = self.integration.create_general_purpose_dataset(
                output_path=f"{output_dir}/base_queries.jsonl"
            )
        
        print(f"✓ Base queries: {len(base_queries)}")
        
        # Step 2: Calculate how many variations needed
        n_variations = (target_size // len(base_queries)) + 1
        print(f"\nStep 2: Generating {n_variations} variations...")
        
        all_queries = self.generate_query_variations(base_queries, n_variations)
        all_queries = all_queries[:target_size]  # Trim to exact target
        
        print(f"✓ Total queries to process: {len(all_queries):,}")
        
        # Step 3: Generate dataset in batches
        print(f"\nStep 3: Querying teacher model...")
        print(f"Estimated time: {len(all_queries) * 0.2 / 60:.1f} minutes")
        print(f"(~12 seconds per query)")
        
        output_file = self.generate_batch(
            queries=all_queries,
            batch_size=100,
            output_dir=output_dir,
            checkpoint_interval=1000
        )
        
        return output_file


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate Large-Scale TRD Dataset (100k+ examples)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100k physics examples
  python trd_framework/generate_large_dataset.py \\
      --mode physics \\
      --target-size 100000 \\
      --teacher-model qwen3:30b-a3b

  # Generate 50k general examples
  python trd_framework/generate_large_dataset.py \\
      --mode general \\
      --target-size 50000 \\
      --teacher-model qwen3:30b-a3b

  # Resume interrupted generation
  python trd_framework/generate_large_dataset.py \\
      --mode physics \\
      --target-size 100000 \\
      --resume
        """
    )
    
    parser.add_argument('--mode', choices=['physics', 'general'], default='physics',
                       help='Dataset mode')
    parser.add_argument('--target-size', type=int, default=100000,
                       help='Target number of examples (default: 100000)')
    parser.add_argument('--teacher-model', default='qwen3:30b-a3b',
                       help='Teacher model name')
    parser.add_argument('--teacher-url', default='http://localhost:11434',
                       help='Ollama API URL')
    parser.add_argument('--output-dir', default='trd_framework/large_dataset',
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Estimate time
    estimated_hours = (args.target_size * 12) / 3600  # 12 seconds per query
    
    print(f"\n{'='*70}")
    print(f"LARGE-SCALE TRD DATASET GENERATION")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Target Size: {args.target_size:,} examples")
    print(f"  Teacher Model: {args.teacher_model}")
    print(f"  Output: {args.output_dir}")
    print(f"\nEstimated Time: {estimated_hours:.1f} hours")
    print(f"  (Can be paused and resumed)")
    print(f"{'='*70}\n")
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Initialize generator
    generator = LargeScaleDatasetGenerator(
        teacher_url=args.teacher_url,
        teacher_model=args.teacher_model,
        mode=args.mode
    )
    
    # Generate dataset
    output_file = generator.generate_large_dataset(
        target_size=args.target_size,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*70}")
    print(f"SUCCESS!")
    print(f"{'='*70}")
    print(f"Dataset: {output_file}")
    print(f"\nNext step: Train student model")
    print(f"  python trd_framework/run_phase2_training.py \\")
    print(f"      --dataset {output_file} \\")
    print(f"      --output-dir trd_framework/models/qwen2-1.5b-large \\")
    print(f"      --epochs 3")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
