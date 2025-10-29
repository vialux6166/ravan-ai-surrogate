#!/usr/bin/env python3
"""Check available Ollama models"""

import requests
import json

try:
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])
        
        print("\n" + "="*70)
        print("AVAILABLE OLLAMA MODELS")
        print("="*70)
        
        qwen_models = []
        other_models = []
        
        for model in models:
            name = model['name']
            size = model.get('size', 0) / (1024**3)  # Convert to GB
            param_size = model.get('details', {}).get('parameter_size', 'Unknown')
            
            if 'qwen' in name.lower():
                qwen_models.append((name, size, param_size))
            else:
                other_models.append((name, size, param_size))
        
        if qwen_models:
            print("\nQwen Models (Recommended for TRD):")
            for name, size, params in qwen_models:
                print(f"  - {name:40s} | {size:6.2f} GB | {params}")
        
        print("\nOther Models:")
        for name, size, params in other_models[:5]:  # Show first 5
            print(f"  - {name:40s} | {size:6.2f} GB | {params}")
        
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        
        if qwen_models:
            best_model = qwen_models[0][0]  # First qwen model
            print(f"\nUse this model for TRD Phase 1:")
            print(f"  {best_model}")
            print(f"\nCommand:")
            print(f"  python trd_framework/run_phase1_complete.py \\")
            print(f"      --mode physics \\")
            print(f"      --teacher-model {best_model}")
        else:
            print("\nNo Qwen models found. Install one with:")
            print("  ollama pull qwen2.5:32b")
            print("  OR")
            print("  ollama pull qwen2.5:14b")
        
        print("="*70 + "\n")
        
    else:
        print("Error: Could not connect to Ollama")
        print("Make sure Ollama is running: ollama serve")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure Ollama is running:")
    print("  ollama serve")
