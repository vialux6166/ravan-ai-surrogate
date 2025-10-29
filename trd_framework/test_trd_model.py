#!/usr/bin/env python3
"""
Test TRD-Trained Model

Simple script to test the trained student model interactively.
"""

import sys
sys.path.insert(0, '.')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def load_trd_model(model_path: str, base_model: str = "Qwen/Qwen2-1.5B-Instruct"):
    """Load TRD-trained model"""
    print(f"\nLoading TRD model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    return model, tokenizer


def generate_response(model, tokenizer, query: str, max_length: int = 1024, temperature: float = 0.7):
    """Generate response from TRD model"""
    # Format input
    input_text = f"Query: {query}\n\nResponse:"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(
        outputs[0][len(inputs['input_ids'][0]):],
        skip_special_tokens=True
    )
    
    return response.strip()


def interactive_mode(model, tokenizer):
    """Interactive testing mode"""
    print("="*70)
    print("TRD MODEL INTERACTIVE TEST")
    print("="*70)
    print("\nEnter your queries (or 'quit' to exit)")
    print("The model will respond with structured reasoning\n")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, query)
            
            print("\n" + "="*70)
            print("RESPONSE:")
            print("="*70)
            print(response)
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def test_examples(model, tokenizer, mode: str = "physics"):
    """Test with example queries"""
    print("="*70)
    print("TRD MODEL TEST - EXAMPLE QUERIES")
    print("="*70)
    
    if mode == "physics":
        examples = [
            "Explain quantum tunneling through a potential barrier.",
            "What is the Von Neumann entropy of a Bell state?",
            "Derive the energy eigenvalues of a quantum harmonic oscillator."
        ]
    else:
        examples = [
            "Solve the integral ∫(x² + 3x - 2)dx",
            "Explain the difference between deep copy and shallow copy in Python.",
            "What is the Heisenberg uncertainty principle?"
        ]
    
    for i, query in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}/{len(examples)}")
        print(f"{'='*70}")
        print(f"Query: {query}\n")
        
        response = generate_response(model, tokenizer, query)
        
        print("Response:")
        print("-"*70)
        print(response)
        print("="*70)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Test TRD-Trained Model')
    parser.add_argument('--model-path', required=True, help='Path to trained TRD model')
    parser.add_argument('--base-model', default='Qwen/Qwen2-1.5B-Instruct',
                       help='Base model name')
    parser.add_argument('--mode', choices=['interactive', 'examples'], default='interactive',
                       help='Test mode: interactive or examples')
    parser.add_argument('--domain', choices=['physics', 'general'], default='physics',
                       help='Domain for example queries')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_trd_model(args.model_path, args.base_model)
    
    # Run test mode
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer)
    else:
        test_examples(model, tokenizer, args.domain)


if __name__ == '__main__':
    main()
