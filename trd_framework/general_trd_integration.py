#!/usr/bin/env python3
"""
General-Purpose TRD Integration

A domain-agnostic implementation of Topological Resonance Distillation.
This creates a general reasoning model that works across multiple domains:
- Mathematics
- Programming
- Science
- Logic & Philosophy
- Creative Writing
- Business & Finance
- Medicine & Health
- Engineering
"""

import json
from pathlib import Path
from typing import List, Dict


class GeneralTRDIntegration:
    """
    General-purpose TRD integration for multi-domain reasoning
    
    This class generates diverse queries across multiple domains to create
    a general-purpose reasoning model, not specialized to any single field.
    """
    
    def __init__(self):
        """Initialize general TRD integration"""
        print("\n" + "="*70)
        print("GENERAL-PURPOSE TRD INTEGRATION")
        print("="*70)
        print("\nCreating multi-domain reasoning dataset")
        print("Domains: Math, Programming, Science, Logic, Creative, Business, Medicine, Engineering")
    
    def generate_mathematics_queries(self) -> List[Dict]:
        """Generate mathematics reasoning queries"""
        return [
            {
                'query': "Solve the integral ∫(x² + 3x - 2)dx and explain each step of your solution.",
                'domain': 'mathematics',
                'subdomain': 'calculus',
                'difficulty': 'intermediate',
                'concepts': ['integration', 'polynomial', 'antiderivative']
            },
            {
                'query': "Prove that the square root of 2 is irrational using proof by contradiction.",
                'domain': 'mathematics',
                'subdomain': 'number_theory',
                'difficulty': 'advanced',
                'concepts': ['proof', 'irrational_numbers', 'contradiction']
            },
            {
                'query': "Find the eigenvalues and eigenvectors of the matrix [[2, 1], [1, 2]].",
                'domain': 'mathematics',
                'subdomain': 'linear_algebra',
                'difficulty': 'intermediate',
                'concepts': ['eigenvalues', 'eigenvectors', 'matrices']
            },
            {
                'query': "Solve the differential equation dy/dx = 2xy with initial condition y(0) = 1.",
                'domain': 'mathematics',
                'subdomain': 'differential_equations',
                'difficulty': 'advanced',
                'concepts': ['ode', 'separation_of_variables', 'initial_value_problem']
            },
            {
                'query': "Explain the Fundamental Theorem of Calculus and why it connects differentiation and integration.",
                'domain': 'mathematics',
                'subdomain': 'calculus',
                'difficulty': 'intermediate',
                'concepts': ['fundamental_theorem', 'integration', 'differentiation']
            }
        ]
    
    def generate_programming_queries(self) -> List[Dict]:
        """Generate programming and computer science queries"""
        return [
            {
                'query': "Debug this Python function:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\nExplain the performance issue and provide an optimized solution.",
                'domain': 'programming',
                'subdomain': 'algorithms',
                'difficulty': 'intermediate',
                'concepts': ['recursion', 'dynamic_programming', 'memoization', 'time_complexity']
            },
            {
                'query': "Explain the difference between deep copy and shallow copy in Python, with examples of when each would cause problems.",
                'domain': 'programming',
                'subdomain': 'python',
                'difficulty': 'intermediate',
                'concepts': ['memory_management', 'references', 'copying']
            },
            {
                'query': "Design a database schema for an e-commerce system with users, products, orders, and reviews. Explain your normalization decisions.",
                'domain': 'programming',
                'subdomain': 'databases',
                'difficulty': 'advanced',
                'concepts': ['database_design', 'normalization', 'relationships', 'sql']
            },
            {
                'query': "Implement a binary search tree in Python with insert, search, and delete operations. Explain the time complexity of each.",
                'domain': 'programming',
                'subdomain': 'data_structures',
                'difficulty': 'advanced',
                'concepts': ['bst', 'tree_traversal', 'time_complexity']
            },
            {
                'query': "Explain how garbage collection works in Python and when you might encounter memory leaks despite having automatic memory management.",
                'domain': 'programming',
                'subdomain': 'python',
                'difficulty': 'advanced',
                'concepts': ['garbage_collection', 'reference_counting', 'memory_leaks']
            }
        ]
    
    def generate_science_queries(self) -> List[Dict]:
        """Generate general science queries (biology, chemistry, physics)"""
        return [
            {
                'query': "Explain the process of photosynthesis at the molecular level, including the light-dependent and light-independent reactions.",
                'domain': 'science',
                'subdomain': 'biology',
                'difficulty': 'intermediate',
                'concepts': ['photosynthesis', 'biochemistry', 'cellular_processes']
            },
            {
                'query': "Why does ice float on water? Explain in terms of molecular structure and hydrogen bonding.",
                'domain': 'science',
                'subdomain': 'chemistry',
                'difficulty': 'intermediate',
                'concepts': ['molecular_structure', 'hydrogen_bonding', 'density']
            },
            {
                'query': "Explain how CRISPR-Cas9 gene editing works and discuss its potential applications and ethical concerns.",
                'domain': 'science',
                'subdomain': 'biology',
                'difficulty': 'advanced',
                'concepts': ['genetics', 'gene_editing', 'biotechnology', 'ethics']
            },
            {
                'query': "Describe the mechanism of enzyme catalysis and explain how temperature and pH affect enzyme activity.",
                'domain': 'science',
                'subdomain': 'biochemistry',
                'difficulty': 'intermediate',
                'concepts': ['enzymes', 'catalysis', 'kinetics', 'protein_structure']
            },
            {
                'query': "Explain the greenhouse effect and how different gases contribute to global warming. Include the molecular basis.",
                'domain': 'science',
                'subdomain': 'environmental_science',
                'difficulty': 'intermediate',
                'concepts': ['climate_change', 'molecular_vibrations', 'infrared_absorption']
            }
        ]
    
    def generate_logic_philosophy_queries(self) -> List[Dict]:
        """Generate logic and philosophical reasoning queries"""
        return [
            {
                'query': "Analyze this argument: 'All humans are mortal. Socrates is human. Therefore, Socrates is mortal.' Identify the logical form and explain why it's valid.",
                'domain': 'logic',
                'subdomain': 'formal_logic',
                'difficulty': 'intermediate',
                'concepts': ['syllogism', 'deductive_reasoning', 'validity']
            },
            {
                'query': "Explain the Trolley Problem and discuss what it reveals about consequentialist vs. deontological ethics.",
                'domain': 'philosophy',
                'subdomain': 'ethics',
                'difficulty': 'advanced',
                'concepts': ['moral_philosophy', 'consequentialism', 'deontology', 'thought_experiments']
            },
            {
                'query': "What is the difference between correlation and causation? Provide examples where confusing them leads to incorrect conclusions.",
                'domain': 'logic',
                'subdomain': 'critical_thinking',
                'difficulty': 'intermediate',
                'concepts': ['causation', 'correlation', 'logical_fallacies']
            },
            {
                'query': "Explain Gödel's Incompleteness Theorems in simple terms and discuss their implications for mathematics and computation.",
                'domain': 'logic',
                'subdomain': 'mathematical_logic',
                'difficulty': 'advanced',
                'concepts': ['incompleteness', 'formal_systems', 'computability']
            }
        ]
    
    def generate_creative_writing_queries(self) -> List[Dict]:
        """Generate creative writing and literature queries"""
        return [
            {
                'query': "Write a compelling opening paragraph for a science fiction story set on a space station. Establish mood, setting, and hint at conflict.",
                'domain': 'creative_writing',
                'subdomain': 'fiction',
                'difficulty': 'intermediate',
                'concepts': ['narrative_hooks', 'world_building', 'atmosphere']
            },
            {
                'query': "Analyze the use of symbolism in this sentence: 'The old oak tree stood alone in the field, its branches reaching toward the storm clouds like desperate fingers.' Explain what makes it effective.",
                'domain': 'creative_writing',
                'subdomain': 'literary_analysis',
                'difficulty': 'intermediate',
                'concepts': ['symbolism', 'imagery', 'metaphor']
            },
            {
                'query': "Improve this dialogue to sound more natural:\n'Hello, John. How are you today?'\n'I am fine, thank you. How are you?'\n'I am also fine. Would you like to go to the store?'",
                'domain': 'creative_writing',
                'subdomain': 'dialogue',
                'difficulty': 'intermediate',
                'concepts': ['dialogue', 'natural_speech', 'character_voice']
            },
            {
                'query': "Create a plot twist for a mystery story where the detective discovers they've been investigating the wrong crime all along. Make it surprising but logical.",
                'domain': 'creative_writing',
                'subdomain': 'plot_structure',
                'difficulty': 'advanced',
                'concepts': ['plot_twist', 'foreshadowing', 'narrative_structure']
            }
        ]
    
    def generate_business_finance_queries(self) -> List[Dict]:
        """Generate business and finance queries"""
        return [
            {
                'query': "A company has revenue of $1M, COGS of $400K, operating expenses of $300K, and interest expense of $50K. Calculate EBITDA, EBIT, and net income. Explain what each metric tells us.",
                'domain': 'business',
                'subdomain': 'finance',
                'difficulty': 'intermediate',
                'concepts': ['financial_metrics', 'income_statement', 'profitability']
            },
            {
                'query': "Explain the concept of Net Present Value (NPV) and why it's important for investment decisions. Include a simple example.",
                'domain': 'business',
                'subdomain': 'finance',
                'difficulty': 'intermediate',
                'concepts': ['npv', 'time_value_of_money', 'investment_analysis']
            },
            {
                'query': "A startup is deciding between bootstrapping and venture capital funding. Analyze the pros and cons of each approach.",
                'domain': 'business',
                'subdomain': 'entrepreneurship',
                'difficulty': 'intermediate',
                'concepts': ['funding', 'equity', 'growth_strategy']
            },
            {
                'query': "Explain Porter's Five Forces framework and apply it to analyze the competitive landscape of the smartphone industry.",
                'domain': 'business',
                'subdomain': 'strategy',
                'difficulty': 'advanced',
                'concepts': ['competitive_analysis', 'strategy', 'market_forces']
            }
        ]
    
    def generate_medicine_health_queries(self) -> List[Dict]:
        """Generate medicine and health queries"""
        return [
            {
                'query': "Explain how vaccines work at the immunological level. Include both the innate and adaptive immune responses.",
                'domain': 'medicine',
                'subdomain': 'immunology',
                'difficulty': 'intermediate',
                'concepts': ['vaccines', 'immune_system', 'antibodies', 'memory_cells']
            },
            {
                'query': "A patient presents with chest pain, shortness of breath, and elevated troponin levels. Walk through the differential diagnosis and explain your reasoning.",
                'domain': 'medicine',
                'subdomain': 'cardiology',
                'difficulty': 'advanced',
                'concepts': ['differential_diagnosis', 'cardiac_markers', 'clinical_reasoning']
            },
            {
                'query': "Explain the mechanism of action of beta-blockers and why they're used to treat hypertension and heart failure.",
                'domain': 'medicine',
                'subdomain': 'pharmacology',
                'difficulty': 'intermediate',
                'concepts': ['pharmacology', 'cardiovascular', 'drug_mechanisms']
            },
            {
                'query': "Describe the pathophysiology of Type 2 diabetes, including insulin resistance and beta-cell dysfunction.",
                'domain': 'medicine',
                'subdomain': 'endocrinology',
                'difficulty': 'advanced',
                'concepts': ['diabetes', 'metabolism', 'pathophysiology']
            }
        ]
    
    def generate_engineering_queries(self) -> List[Dict]:
        """Generate engineering queries"""
        return [
            {
                'query': "Design a simple bridge to span 50 meters. Explain your choice of bridge type, materials, and key structural considerations.",
                'domain': 'engineering',
                'subdomain': 'civil_engineering',
                'difficulty': 'advanced',
                'concepts': ['structural_design', 'materials', 'load_analysis']
            },
            {
                'query': "Explain how a transistor works and why it's fundamental to modern computing. Include both the physics and the logic gate perspective.",
                'domain': 'engineering',
                'subdomain': 'electrical_engineering',
                'difficulty': 'intermediate',
                'concepts': ['semiconductors', 'transistors', 'digital_logic']
            },
            {
                'query': "A heat exchanger needs to cool 1000 kg/hr of water from 80°C to 40°C. Explain the design considerations and calculate the heat transfer rate.",
                'domain': 'engineering',
                'subdomain': 'mechanical_engineering',
                'difficulty': 'advanced',
                'concepts': ['heat_transfer', 'thermodynamics', 'design']
            },
            {
                'query': "Explain the trade-offs between microservices and monolithic architecture for a web application. When would you choose each?",
                'domain': 'engineering',
                'subdomain': 'software_engineering',
                'difficulty': 'advanced',
                'concepts': ['architecture', 'scalability', 'design_patterns']
            }
        ]
    
    def create_general_purpose_dataset(
        self,
        output_path: str = "trd_framework/general_purpose_queries.jsonl",
        queries_per_domain: int = None
    ) -> List[Dict]:
        """
        Create a complete general-purpose query dataset
        
        This combines queries from all domains to create a multi-domain
        reasoning model.
        
        Args:
            output_path: Where to save the dataset
            queries_per_domain: Number of queries per domain (None = all)
            
        Returns:
            List of all queries
        """
        print("\n" + "="*70)
        print("CREATING GENERAL-PURPOSE QUERY DATASET")
        print("="*70)
        
        all_queries = []
        
        # Generate queries from all domains
        domain_generators = {
            'Mathematics': self.generate_mathematics_queries,
            'Programming': self.generate_programming_queries,
            'Science': self.generate_science_queries,
            'Logic & Philosophy': self.generate_logic_philosophy_queries,
            'Creative Writing': self.generate_creative_writing_queries,
            'Business & Finance': self.generate_business_finance_queries,
            'Medicine & Health': self.generate_medicine_health_queries,
            'Engineering': self.generate_engineering_queries
        }
        
        for domain_name, generator in domain_generators.items():
            print(f"\n{domain_name}...")
            queries = generator()
            
            if queries_per_domain:
                queries = queries[:queries_per_domain]
            
            all_queries.extend(queries)
            print(f"  ✓ Generated {len(queries)} queries")
        
        print(f"\n{'='*70}")
        print(f"Total queries: {len(all_queries)}")
        
        # Print domain distribution
        domain_counts = {}
        for q in all_queries:
            domain = q['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print("\nDomain Distribution:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count} queries")
        
        # Save dataset
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for query in all_queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        print(f"\n✓ General-purpose query dataset saved to: {output_path}")
        print(f"\nNext step: Use these queries with teacher model to generate reasoning dataset")
        print(f"  python trd_framework/run_phase1_complete.py --mode general")
        
        return all_queries


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("GENERAL-PURPOSE TRD INTEGRATION")
    print("="*70)
    print("\nCreating a multi-domain reasoning model")
    
    # Initialize integration
    integration = GeneralTRDIntegration()
    
    # Create general-purpose query dataset
    queries = integration.create_general_purpose_dataset()
    
    print(f"\n✓ Integration complete!")
    print(f"  Query dataset size: {len(queries)} examples")
    print(f"  Domains covered: 8")
    print(f"\nThis dataset can be used to train a general-purpose reasoning model")
    print(f"that works across mathematics, programming, science, and more!")
    
    return queries


if __name__ == '__main__':
    queries = main()
