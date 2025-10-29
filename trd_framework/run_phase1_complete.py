#!/usr/bin/env python3
"""
TRD Phase 1 Complete: Resonance Dataset Generation

This script supports two modes:
1. PHYSICS MODE: Physics-grounded queries from Ravan simulations
2. GENERAL MODE: Multi-domain queries for general-purpose reasoning

Both modes:
- Query the teacher model (Qwen 32B) with chain-of-thought prompts
- Create a complete resonance dataset for TRD training
"""

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from typing import List, Dict

# Import TRD components
from trd_framework.phase1_dataset_generation import ResonanceDatasetGenerator
from trd_framework.ravan_trd_integration import RavanTRDIntegration
from trd_framework.general_trd_integration import GeneralTRDIntegration


class TRDPhase1Complete:
    """
    Complete Phase 1 pipeline:
    Query Generation → Teacher Queries → Resonance Dataset
    
    Supports two modes:
    - physics: Ravan quantum physics queries
    - general: Multi-domain general reasoning queries
    """
    
    def __init__(self, teacher_url: str = "http://localhost:11434", mode: str = "physics"):
        """
        Initialize Phase 1 pipeline
        
        Args:
            teacher_url: URL for teacher model API (Ollama)
            mode: 'physics' for Ravan-specific or 'general' for multi-domain
        """
        self.teacher_url = teacher_url
        self.mode = mode
        self.dataset_generator = ResonanceDatasetGenerator(teacher_url)
        
        if mode == "physics":
            self.integration = RavanTRDIntegration()
        elif mode == "general":
            self.integration = GeneralTRDIntegration()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'physics' or 'general'")
    
    def run_complete_phase1(
        self,
        n_simulation_queries: int = 60,
        teacher_model: str = "qwen2.5:32b",
        output_dir: str = "trd_framework"
    ) -> Dict:
        """
        Run complete Phase 1 pipeline
        
        Args:
            n_simulation_queries: Number of simulation-based queries (physics mode only)
            teacher_model: Teacher model name
            output_dir: Output directory
            
        Returns:
            Dictionary with paths and statistics
        """
        print("\n" + "="*80)
        print("TRD PHASE 1 COMPLETE: RESONANCE DATASET GENERATION")
        print("="*80)
        print(f"\nMode: {self.mode.upper()}")
        print(f"Teacher Model: {teacher_model}")
        print(f"Output Directory: {output_dir}")
        
        # Step 1: Generate queries based on mode
        print("\n" + "="*80)
        if self.mode == "physics":
            print("STEP 1: GENERATE PHYSICS-SPECIFIC QUERIES")
            print("="*80)
            queries_path = f"{output_dir}/ravan_physics_queries.jsonl"
            query_list = self.integration.create_ravan_specific_dataset(
                output_path=queries_path,
                n_simulation_queries=n_simulation_queries
            )
        else:  # general mode
            print("STEP 1: GENERATE GENERAL-PURPOSE QUERIES")
            print("="*80)
            queries_path = f"{output_dir}/general_purpose_queries.jsonl"
            query_list = self.integration.create_general_purpose_dataset(
                output_path=queries_path
            )
        
        # Load queries
        queries_text = [q['query'] for q in query_list]
        
        print(f"\n✓ Step 1 Complete: {len(queries_text)} queries generated")
        
        # Step 2: Query teacher model with chain-of-thought prompts
        print("\n" + "="*80)
        print("STEP 2: QUERY TEACHER MODEL")
        print("="*80)
        print(f"\nThis will query {teacher_model} for each question...")
        print("Note: This requires Ollama running with the teacher model")
        print("Install: curl -fsSL https://ollama.com/install.sh | sh")
        print(f"Run: ollama run {teacher_model}")
        
        # Check if we should proceed
        print(f"\nReady to query teacher model?")
        print(f"This will take approximately {len(queries_text) * 0.15:.0f}-{len(queries_text) * 0.2:.0f} minutes for {len(queries_text)} queries.")
        
        if self.mode == "physics":
            resonance_path = f"{output_dir}/ravan_resonance_dataset.jsonl"
            merged_path = f"{output_dir}/ravan_trd_complete_dataset.jsonl"
        else:
            resonance_path = f"{output_dir}/general_resonance_dataset.jsonl"
            merged_path = f"{output_dir}/general_trd_complete_dataset.jsonl"
        
        try:
            # Generate resonance dataset
            resonance_examples = self.dataset_generator.generate_dataset(
                queries=queries_text,
                output_path=resonance_path,
                teacher_model=teacher_model
            )
            
            print(f"\n✓ Step 2 Complete: {len(resonance_examples)} reasoning examples generated")
            
            # Step 3: Merge query metadata with resonance examples
            print("\n" + "="*80)
            print("STEP 3: MERGE METADATA")
            print("="*80)
            
            merged_examples = self._merge_datasets(query_list, resonance_examples, merged_path)
            
            print(f"\n✓ Step 3 Complete: {len(merged_examples)} complete examples")
            
            # Summary
            results = {
                'queries_path': queries_path,
                'resonance_path': resonance_path,
                'merged_path': merged_path,
                'total_examples': len(merged_examples),
                'teacher_model': teacher_model,
                'status': 'success'
            }
            
            print("\n" + "="*80)
            print("PHASE 1 COMPLETE!")
            print("="*80)
            print(f"\nGenerated Files:")
            print(f"  1. Queries: {queries_path}")
            print(f"  2. Resonance: {resonance_path}")
            print(f"  3. Complete: {merged_path}")
            print(f"\nTotal Examples: {len(merged_examples)}")
            print(f"\nNext Step: Run Phase 2 (Student Model Training)")
            print(f"  python trd_framework/run_phase2_training.py")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Error during teacher querying: {e}")
            print("\nMake sure Ollama is running:")
            print(f"  ollama run {teacher_model}")
            
            return {
                'queries_path': queries_path,
                'total_queries': len(queries_text),
                'status': 'failed',
                'error': str(e)
            }
    
    def _merge_datasets(
        self,
        query_list: List[Dict],
        resonance_examples: List[Dict],
        output_path: str
    ) -> List[Dict]:
        """
        Merge query metadata with resonance examples
        
        Args:
            query_list: Original queries with metadata
            resonance_examples: Resonance examples from teacher
            output_path: Where to save merged dataset
            
        Returns:
            List of merged examples
        """
        merged = []
        
        for i, (query_data, resonance) in enumerate(zip(query_list, resonance_examples)):
            merged_example = {
                **resonance,  # Include all resonance data
                'simulation_type': query_data.get('simulation_type', 'unknown'),
                'parameters': query_data.get('parameters', {}),
                'ground_truth': query_data.get('ground_truth', {}),
                'physics_concepts': query_data.get('physics_concepts', [])
            }
            merged.append(merged_example)
        
        # Save merged dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in merged:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        return merged


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TRD Phase 1: Complete Resonance Dataset Generation')
    parser.add_argument('--mode', choices=['physics', 'general'], default='physics',
                       help='Mode: physics (Ravan-specific) or general (multi-domain)')
    parser.add_argument('--teacher-url', default='http://localhost:11434', help='Ollama API URL')
    parser.add_argument('--teacher-model', default='qwen2.5:32b', help='Teacher model name')
    parser.add_argument('--n-queries', type=int, default=60, 
                       help='Number of simulation queries (physics mode only)')
    parser.add_argument('--output-dir', default='trd_framework', help='Output directory')
    
    args = parser.parse_args()
    
    # Run Phase 1
    phase1 = TRDPhase1Complete(teacher_url=args.teacher_url, mode=args.mode)
    results = phase1.run_complete_phase1(
        n_simulation_queries=args.n_queries,
        teacher_model=args.teacher_model,
        output_dir=args.output_dir
    )
    
    return results


if __name__ == '__main__':
    results = main()
