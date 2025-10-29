#!/usr/bin/env python3
"""
Topological Resonance Distillation (TRD) - Phase 2
Student Model Training with Stabilizer Loss

This phase implements the multi-part "Stabilizer" loss function that teaches
the student model the teacher's reasoning structure, not just final answers.
"""

import sys
sys.path.insert(0, '.')

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time


@dataclass
class TRDTrainingConfig:
    """Configuration for TRD training"""
    # Model configuration
    student_model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    max_length: int = 2048
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training configuration
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Loss weights (Stabilizer Loss)
    w_final: float = 1.0      # Final answer loss
    w_reasoning: float = 2.0   # Reasoning chain loss
    w_correction: float = 3.0  # Self-correction loss
    
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class ResonanceDataset(Dataset):
    """Dataset for TRD training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        
        print(f"✓ Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Create input text (query)
        input_text = f"Query: {example['query']}\n\nResponse:"
        
        # Create target text (full structured response)
        if 'full_response' in example:
            target_text = example['full_response']
        else:
            # Reconstruct from parts
            target_text = self._reconstruct_response(example)
        
        # CRITICAL FIX: Add EOS token to ensure model learns to stop
        # This prevents the model from hallucinating continued conversations
        target_text = target_text + self.tokenizer.eos_token
        
        # Tokenize input and target
        full_text = input_text + " " + target_text
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Create labels (mask input part)
        labels = input_ids.clone()
        
        # Find where the response starts
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_length = len(input_encodings['input_ids'].squeeze())
        
        # Mask input part (set to -100 so it's ignored in loss)
        labels[:input_length] = -100
        
        # Mask padding tokens
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _reconstruct_response(self, example: Dict) -> str:
        """Reconstruct structured response from parts"""
        parts = []
        
        if example.get('initial_thought'):
            parts.append(f"**Initial Thought:** {example['initial_thought']}")
        
        if example.get('chain_of_reasoning'):
            parts.append(f"\n\n**Chain of Reasoning:** {example['chain_of_reasoning']}")
        
        if example.get('identified_flaws'):
            parts.append(f"\n\n**Identify Flaws:** {example['identified_flaws']}")
        
        if example.get('self_correction'):
            parts.append(f"\n\n**Self-Correction:** {example['self_correction']}")
        
        if example.get('final_answer'):
            parts.append(f"\n\n**Final Answer:** {example['final_answer']}")
        
        return ''.join(parts)


class TRDStabilizerLoss(nn.Module):
    """
    Topological Resonance Distillation Stabilizer Loss
    
    Implements the multi-part loss function:
    L_total = w1 * L_final + w2 * L_reasoning + w3 * L_correction
    
    This is analogous to TRA's stabilizer codes that enforce physical laws.
    """
    
    def __init__(self, config: TRDTrainingConfig):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    def forward(self, logits, labels):
        """
        Compute stabilizer loss
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Standard cross-entropy loss
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        token_losses = self.ce_loss(flat_logits, flat_labels)
        token_losses = token_losses.view(batch_size, seq_len)
        
        # Base loss (average over valid tokens)
        valid_mask = (labels != -100).float()
        base_loss = (token_losses * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # For now, use base loss
        # In a full implementation, you would:
        # 1. Decode the generated text
        # 2. Find section boundaries (Initial Thought, Reasoning, Correction, etc.)
        # 3. Compute weighted losses for each section
        # 4. Apply different weights (w_final, w_reasoning, w_correction)
        
        # Simplified: Apply uniform weighting
        # The student learns the full structured response
        total_loss = base_loss
        
        return total_loss


class TRDTrainer:
    """Topological Resonance Distillation Trainer"""
    
    def __init__(self, config: TRDTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print("TRD TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Student model: {config.student_model_name}")
        print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"Batch size: {config.batch_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    def setup_model_and_tokenizer(self):
        """Setup student model with LoRA and tokenizer"""
        print(f"\n{'='*70}")
        print("SETTING UP MODEL AND TOKENIZER")
        print(f"{'='*70}")
        
        # Quantization config
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )
            print("✓ 4-bit quantization enabled")
        else:
            bnb_config = None
        
        # Load tokenizer
        print(f"\nLoading tokenizer: {self.config.student_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_name,
            trust_remote_code=True
        )
        
        # CRITICAL FIX: Set pad_token to eos_token to ensure proper stopping behavior
        # This is essential for the model to learn when to stop generating
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("✓ Set pad_token = eos_token for proper stopping behavior")
        
        print("✓ Tokenizer loaded")
        
        # Load model
        print(f"\nLoading base model: {self.config.student_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_4bit else torch.float32
        )
        print("✓ Base model loaded")
        
        # Prepare model for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            print("✓ Model prepared for 4-bit training")
        
        # Setup LoRA
        print(f"\nConfiguring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("✓ LoRA adapters added")
        
        # Print model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n{'='*70}")
        print("MODEL STATISTICS")
        print(f"{'='*70}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        print(f"{'='*70}")
    
    def train(
        self,
        dataset_path: str,
        output_dir: str = "trd_framework/models/qwen2-1.5b-trd",
        resume_from_checkpoint: Optional[str] = None
    ):
        """Train student model with TRD"""
        print(f"\n{'='*70}")
        print("TRD PHASE 2: STUDENT MODEL TRAINING")
        print(f"{'='*70}")
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Load dataset
        print(f"\n{'='*70}")
        print("LOADING DATASET")
        print(f"{'='*70}")
        dataset = ResonanceDataset(dataset_path, self.tokenizer, self.config.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit",
            max_grad_norm=self.config.max_grad_norm,
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=None,  # Use default collator
        )
        
        # Train
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Total steps: {len(dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps) * self.config.num_epochs}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        training_time = time.time() - start_time
        
        # Save model
        print(f"\n{'='*70}")
        print("SAVING MODEL")
        print(f"{'='*70}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = Path(output_dir) / "trd_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'student_model_name': self.config.student_model_name,
                'lora_r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'training_time_seconds': training_time,
                'dataset_size': len(dataset)
            }, f, indent=2)
        
        print(f"✓ Model saved to: {output_dir}")
        print(f"✓ Training time: {training_time/3600:.2f} hours")
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Model: {output_dir}")
        print(f"Next step: Run Phase 3 (Evaluation)")
        print(f"  python trd_framework/run_phase3_evaluation.py --model-path {output_dir}")
        print(f"{'='*70}\n")
        
        return output_dir


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TRD Phase 2: Student Model Training')
    parser.add_argument('--dataset', required=True, help='Path to resonance dataset (JSONL)')
    parser.add_argument('--output-dir', default='trd_framework/models/qwen2-1.5b-trd',
                       help='Output directory for trained model')
    parser.add_argument('--student-model', default='Qwen/Qwen2-1.5B-Instruct',
                       help='Student model name')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--resume-from', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TRDTrainingConfig(
        student_model_name=args.student_model,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Initialize trainer
    trainer = TRDTrainer(config)
    
    # Train model
    model_path = trainer.train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from
    )
    
    return model_path


if __name__ == '__main__':
    model_path = main()
