#!/usr/bin/env python3
"""
TRD Phase 2: Run Student Model Training

Simple script to train the student model with sensible defaults.
"""

import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path
from trd_framework.phase2_student_training import TRDTrainer, TRDTrainingConfig


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='TRD Phase 2: Train Student Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on physics dataset
  python trd_framework/run_phase2_training.py \\
      --dataset trd_framework/ravan_trd_complete_dataset.jsonl \\
      --output-dir trd_framework/models/qwen2-1.5b-ravan-physics

  # Train on general dataset
  python trd_framework/run_phase2_training.py \\
      --dataset trd_framework/general_trd_complete_dataset.jsonl \\
      --output-dir trd_framework/models/qwen2-1.5b-general

  # Train on hybrid dataset
  python trd_framework/run_phase2_training.py \\
      --dataset trd_framework/hybrid_trd_dataset.jsonl \\
      --output-dir trd_framework/models/qwen2-1.5b-hybrid

  # Custom training parameters
  python trd_framework/run_phase2_training.py \\
      --dataset trd_framework/ravan_trd_complete_dataset.jsonl \\
      --batch-size 1 \\
      --gradient-accumulation 16 \\
      --epochs 5 \\
      --learning-rate 1e-4
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True,
                       help='Path to resonance dataset (JSONL file from Phase 1)')
    
    # Optional arguments
    parser.add_argument('--output-dir', default='trd_framework/models/qwen2-1.5b-trd',
                       help='Output directory for trained model (default: trd_framework/models/qwen2-1.5b-trd)')
    parser.add_argument('--student-model', default='Qwen/Qwen2-1.5B-Instruct',
                       help='Student model name (default: Qwen/Qwen2-1.5B-Instruct)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (default: 2, reduce to 1 if OOM)')
    parser.add_argument('--gradient-accumulation', type=int, default=8,
                       help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    
    # LoRA parameters
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha (default: 16)')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA dropout (default: 0.1)')
    
    # Advanced options
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum sequence length (default: 2048)')
    parser.add_argument('--resume-from', default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--no-4bit', action='store_true',
                       help='Disable 4-bit quantization (requires more VRAM)')
    
    args = parser.parse_args()
    
    # Validate dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found: {args.dataset}")
        print("\nMake sure you've run Phase 1 first:")
        print("  python trd_framework/run_phase1_complete.py --mode physics")
        print("  OR")
        print("  python trd_framework/run_phase1_complete.py --mode general")
        return None
    
    # Print configuration
    print("\n" + "="*70)
    print("TRD PHASE 2: STUDENT MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output_dir}")
    print(f"  Student Model: {args.student_model}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Gradient Accumulation: {args.gradient_accumulation}")
    print(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  4-bit Quantization: {'Disabled' if args.no_4bit else 'Enabled'}")
    print("="*70)
    
    # Create configuration
    config = TRDTrainingConfig(
        student_model_name=args.student_model,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        use_4bit=not args.no_4bit
    )
    
    # Initialize trainer
    trainer = TRDTrainer(config)
    
    # Train model
    try:
        model_path = trainer.train(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from
        )
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nTrained model saved to: {model_path}")
        print(f"\nNext steps:")
        print(f"  1. Evaluate the model:")
        print(f"     python trd_framework/run_phase3_evaluation.py --model-path {model_path}")
        print(f"\n  2. Test the model interactively:")
        print(f"     python trd_framework/test_trd_model.py --model-path {model_path}")
        print(f"\n  3. Deploy to Ravan (if physics mode):")
        print(f"     # Integration with Ravan system")
        print("="*70 + "\n")
        
        return model_path
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR DURING TRAINING")
        print(f"{'='*70}")
        print(f"\nError: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. CUDA Out of Memory:")
        print(f"     - Reduce batch size: --batch-size 1")
        print(f"     - Increase gradient accumulation: --gradient-accumulation 16")
        print(f"     - Reduce max length: --max-length 1024")
        print(f"\n  2. Model loading issues:")
        print(f"     - Check internet connection (downloads model)")
        print(f"     - Verify HuggingFace access")
        print(f"\n  3. Dataset issues:")
        print(f"     - Verify dataset exists: {args.dataset}")
        print(f"     - Check dataset format (JSONL)")
        print(f"     - Ensure Phase 1 completed successfully")
        print("="*70 + "\n")
        
        raise


if __name__ == '__main__':
    model_path = main()
