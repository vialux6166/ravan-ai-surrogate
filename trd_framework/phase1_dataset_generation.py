#!/usr/bin/env python3
"""
Topological Resonance Distillation (TRD) - Phase 1
Dataset Generation: Create "Resonance" Dataset from Teacher Model

This phase generates a dataset that captures the teacher's reasoning process,
not just outputs. The dataset includes chain-of-thought reasoning and self-correction.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import requests


@dataclass
class ReasoningExample:
    """Single reasoning example from teacher model"""
    query: str
    initial_thought: str
    chain_of_reasoning: str
    identified_flaws: str
    self_correction: str
    final_answer: str
    metadata: Dict


class ResonanceDatasetGenerator:
    """
    Generate resonance dataset from teacher model
    
    Uses chain-of-thought prompting to capture teacher's reasoning process
    """
    
    def __init__(self, teacher_model_url: str = "http://localhost:11434"):
        """
        Initialize dataset generator
        
        Args:
            teacher_model_url: URL for teacher model API (e.g., Ollama)
        """
        self.teacher_url = teacher_model_url
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create chain-of-thought prompt template"""
        return """You are a meticulous reasoning engine. For the following query, provide your answer by following these steps:

1. **Initial Thought:** Briefly state your initial hypothesis or approach.
2. **Chain of Reasoning:** Break down the problem and reason through it step-by-step. Show your work.
3. **Identify Flaws:** Actively critique your own chain of reasoning. Are there any logical gaps, potential biases, or missing information?
4. **Self-Correction:** Refine your reasoning based on the identified flaws.
5. **Final Answer:** Provide the final, corrected answer.

**Query:** {query}

Please follow the format exactly, using the section headers as shown above."""
    
    def query_teacher(self, query: str, model: str = "qwen2.5:32b") -> str:
        """
        Query teacher model via Ollama API
        
        Args:
            query: Question to ask
            model: Teacher model name
            
        Returns:
            Teacher's response
        """
        prompt = self.prompt_template.format(query=query)
        
        try:
            response = requests.post(
                f"{self.teacher_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2048
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error querying teacher: {e}")
            return None
    
    def parse_response(self, response: str) -> ReasoningExample:
        """
        Parse teacher's structured response
        
        Args:
            response: Raw response from teacher
            
        Returns:
            Parsed ReasoningExample
        """
        sections = {
            'initial_thought': '',
            'chain_of_reasoning': '',
            'identified_flaws': '',
            'self_correction': '',
            'final_answer': ''
        }
        
        # Simple parsing (can be improved with regex)
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'initial thought' in line_lower:
                current_section = 'initial_thought'
            elif 'chain of reasoning' in line_lower:
                current_section = 'chain_of_reasoning'
            elif 'identify flaws' in line_lower or 'identified flaws' in line_lower:
                current_section = 'identified_flaws'
            elif 'self-correction' in line_lower or 'self correction' in line_lower:
                current_section = 'self_correction'
            elif 'final answer' in line_lower:
                current_section = 'final_answer'
            elif current_section and line.strip():
                sections[current_section] += line + '\n'
        
        return sections
    
    def generate_quantum_physics_queries(self) -> List[str]:
        """Generate quantum physics questions for Ravan domain"""
        return [
            # Quantum Tunneling
            "Explain quantum tunneling through a potential barrier and how barrier height affects transmission coefficient.",
            "What happens to the transmission coefficient when the barrier width increases?",
            "How does the initial momentum of a wavepacket affect its ability to tunnel through a barrier?",
            "Derive the relationship between barrier height V0 and reflection coefficient R.",
            
            # Quantum Circuits
            "Explain how a CNOT gate creates entanglement between two qubits.",
            "What is the Von Neumann entropy of a Bell state and why?",
            "How does measurement collapse affect quantum superposition?",
            "Explain the difference between quantum entanglement and classical correlation.",
            
            # Harmonic Oscillator
            "Derive the energy eigenvalues of a quantum harmonic oscillator.",
            "Why are the energy levels of a quantum harmonic oscillator equally spaced?",
            "Explain the concept of zero-point energy in quantum mechanics.",
            "What is a coherent state and how does it evolve in time?",
            
            # General Quantum Mechanics
            "Explain the Heisenberg uncertainty principle and its physical implications.",
            "What is wavefunction normalization and why is it important?",
            "Explain the difference between stationary and time-dependent Schrödinger equations.",
            "How does the Hamiltonian operator relate to energy in quantum mechanics?",
            
            # ML for Physics
            "How can machine learning accelerate quantum simulations?",
            "What are the challenges of using neural networks to predict quantum observables?",
            "Explain why physics-informed loss functions improve ML model accuracy.",
            "How does uncertainty quantification help validate ML predictions in physics?",
        ]
    
    def generate_dataset(
        self,
        queries: List[str],
        output_path: str = "trd_framework/resonance_dataset.jsonl",
        teacher_model: str = "qwen2.5:32b"
    ):
        """
        Generate complete resonance dataset
        
        Args:
            queries: List of questions
            output_path: Where to save dataset
            teacher_model: Teacher model name
        """
        print("="*70)
        print("TRD PHASE 1: RESONANCE DATASET GENERATION")
        print("="*70)
        print(f"\nTeacher Model: {teacher_model}")
        print(f"Queries: {len(queries)}")
        print(f"Output: {output_path}")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        examples = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing: {query[:60]}...")
            
            # Query teacher
            response = self.query_teacher(query, teacher_model)
            
            if response:
                # Parse response
                sections = self.parse_response(response)
                
                # Create example
                example = {
                    'query': query,
                    'initial_thought': sections['initial_thought'].strip(),
                    'chain_of_reasoning': sections['chain_of_reasoning'].strip(),
                    'identified_flaws': sections['identified_flaws'].strip(),
                    'self_correction': sections['self_correction'].strip(),
                    'final_answer': sections['final_answer'].strip(),
                    'full_response': response,
                    'metadata': {
                        'teacher_model': teacher_model,
                        'timestamp': time.time(),
                        'query_index': i
                    }
                }
                
                examples.append(example)
                
                # Save incrementally
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
                print(f"  ✓ Generated {len(sections['chain_of_reasoning'])} chars of reasoning")
            else:
                print(f"  ✗ Failed to get response")
            
            # Rate limiting
            time.sleep(1)
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  Total examples: {len(examples)}")
        print(f"  Saved to: {output_path}")
        
        return examples


def main():
    """Generate resonance dataset"""
    
    # Initialize generator
    generator = ResonanceDatasetGenerator()
    
    # Generate quantum physics queries
    queries = generator.generate_quantum_physics_queries()
    
    print(f"\nGenerated {len(queries)} quantum physics queries")
    print("\nNote: This requires a teacher model running via Ollama")
    print("Install: curl -fsSL https://ollama.com/install.sh | sh")
    print("Run: ollama run qwen2.5:32b")
    
    # Generate dataset
    dataset = generator.generate_dataset(
        queries=queries,
        output_path="trd_framework/resonance_dataset.jsonl",
        teacher_model="qwen2.5:32b"
    )
    
    return dataset


if __name__ == '__main__':
    dataset = main()
