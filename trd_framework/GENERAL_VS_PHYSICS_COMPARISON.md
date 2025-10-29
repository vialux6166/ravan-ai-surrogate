# TRD Framework: General-Purpose vs Physics-Specific

## Overview

The TRD (Topological Resonance Distillation) framework now supports **two modes**:

1. **Physics Mode**: Specialized for Ravan's quantum physics domain
2. **General Mode**: Multi-domain reasoning across 8 different fields

Both modes use the **same core TRD framework** (chain-of-thought prompts, stabilizer loss, LoRA training), but differ in their query datasets.

## Comparison Table

| Aspect | Physics Mode | General Mode |
|--------|--------------|--------------|
| **Purpose** | Quantum physics expert for Ravan | General-purpose reasoning assistant |
| **Domains** | 1 (Quantum Physics) | 8 (Math, Programming, Science, etc.) |
| **Query Generation** | From real simulations | Curated expert queries |
| **Ground Truth** | Simulation observables | Domain-specific validation |
| **Total Queries** | 69 | 35 |
| **Use Case** | Ravan system integration | Standalone reasoning model |
| **Model Size** | ~3.2GB (Qwen2-1.5B) | ~3.2GB (Qwen2-1.5B) |
| **Training Time** | ~2-4 hours | ~1-2 hours |

## Physics Mode Details

### Query Sources

**Physics-Grounded (60 queries)**:
- **Schrödinger Solver** (20): Quantum tunneling, barrier penetration
- **Quantum Circuit** (20): Entanglement, measurement, entropy
- **Harmonic Oscillator** (20): Energy eigenstates, photon number

**General Quantum (5 queries)**:
- Wavefunction collapse
- Schrödinger equation
- Density matrix formalism
- Uncertainty principle
- Decoherence

**ML for Physics (4 queries)**:
- Physics-informed ML
- Uncertainty quantification
- Conservation laws
- Adaptive sampling

### Example Physics Query

```json
{
  "query": "A quantum wavepacket with initial momentum k₀ = 5.15 encounters a rectangular potential barrier of height V₀ = 7.78 eV and width = 1.03 nm. Explain the quantum tunneling process...",
  "simulation_type": "schrodinger",
  "parameters": {
    "V0": 7.78,
    "barrier_width": 1.03,
    "k0": 5.15
  },
  "ground_truth": {
    "transmission": 0.927,
    "reflection": 0.073
  },
  "physics_concepts": ["quantum_tunneling", "wavefunction"]
}
```

### Strengths

✅ **Physics Expertise**: Deep understanding of quantum mechanics  
✅ **Grounded in Reality**: Queries from actual simulations  
✅ **Verifiable**: Ground truth from physics calculations  
✅ **Ravan Integration**: Seamless deployment to Ravan system  

### Use Cases

- Explaining Ravan simulation results
- Generating quantum physics code
- Teaching quantum mechanics concepts
- Validating ML predictions against physics

## General Mode Details

### Query Sources

**Mathematics (5 queries)**:
- Calculus (integration, fundamental theorem)
- Linear algebra (eigenvalues, eigenvectors)
- Differential equations
- Number theory (proofs)

**Programming (5 queries)**:
- Algorithms (recursion, optimization)
- Data structures (BST, complexity)
- Databases (schema design, normalization)
- Python internals (garbage collection, copying)

**Science (5 queries)**:
- Biology (photosynthesis, CRISPR)
- Chemistry (molecular structure, bonding)
- Biochemistry (enzymes, catalysis)
- Environmental science (climate change)

**Logic & Philosophy (4 queries)**:
- Formal logic (syllogisms, validity)
- Ethics (trolley problem, moral philosophy)
- Critical thinking (correlation vs causation)
- Mathematical logic (Gödel's theorems)

**Creative Writing (4 queries)**:
- Fiction writing (openings, atmosphere)
- Literary analysis (symbolism, metaphor)
- Dialogue improvement
- Plot structure (twists, foreshadowing)

**Business & Finance (4 queries)**:
- Financial metrics (EBITDA, EBIT, NPV)
- Investment analysis (time value of money)
- Entrepreneurship (funding strategies)
- Strategy (Porter's Five Forces)

**Medicine & Health (4 queries)**:
- Immunology (vaccines, immune response)
- Cardiology (differential diagnosis)
- Pharmacology (drug mechanisms)
- Endocrinology (diabetes pathophysiology)

**Engineering (4 queries)**:
- Civil engineering (bridge design)
- Electrical engineering (transistors, logic gates)
- Mechanical engineering (heat transfer)
- Software engineering (architecture patterns)

### Example General Query

```json
{
  "query": "Debug this Python function:\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\nExplain the performance issue and provide an optimized solution.",
  "domain": "programming",
  "subdomain": "algorithms",
  "difficulty": "intermediate",
  "concepts": ["recursion", "dynamic_programming", "memoization", "time_complexity"]
}
```

### Strengths

✅ **Broad Knowledge**: Covers 8 different domains  
✅ **Versatile**: Can handle diverse question types  
✅ **Transferable Skills**: Reasoning patterns apply across fields  
✅ **Standalone**: Works independently of any specific system  

### Use Cases

- General-purpose AI assistant
- Educational tutoring across subjects
- Code review and debugging
- Creative writing assistance
- Business analysis
- Medical education

## Technical Implementation

### Shared Components

Both modes use the **same TRD infrastructure**:

```python
# Chain-of-Thought Prompt Template (Universal)
1. Initial Thought: Hypothesis or approach
2. Chain of Reasoning: Step-by-step analysis
3. Identify Flaws: Self-critique
4. Self-Correction: Refinement
5. Final Answer: Corrected solution

# Stabilizer Loss (Universal)
L_total = w1·L_final + w2·L_reasoning + w3·L_correction

# Training Configuration (Universal)
- Base Model: Qwen2-1.5B-Instruct
- LoRA: r=8, alpha=16
- Quantization: 4-bit
- Batch Size: 2
- Gradient Accumulation: 8
```

### Mode-Specific Components

**Physics Mode**:
```python
# Integration
from trd_framework.ravan_trd_integration import RavanTRDIntegration

# Query Generation
integration = RavanTRDIntegration()
queries = integration.create_ravan_specific_dataset()

# Output
ravan_physics_queries.jsonl (69 queries)
ravan_trd_complete_dataset.jsonl (with teacher responses)
```

**General Mode**:
```python
# Integration
from trd_framework.general_trd_integration import GeneralTRDIntegration

# Query Generation
integration = GeneralTRDIntegration()
queries = integration.create_general_purpose_dataset()

# Output
general_purpose_queries.jsonl (35 queries)
general_trd_complete_dataset.jsonl (with teacher responses)
```

## Usage

### Running Physics Mode

```bash
# Generate physics queries
python trd_framework/ravan_trd_integration.py

# Complete Phase 1 (with teacher model)
python trd_framework/run_phase1_complete.py \
    --mode physics \
    --teacher-model qwen2.5:32b \
    --n-queries 60

# Train student model
python trd_framework/run_phase2_training.py \
    --dataset trd_framework/ravan_trd_complete_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-ravan-physics
```

### Running General Mode

```bash
# Generate general queries
python trd_framework/general_trd_integration.py

# Complete Phase 1 (with teacher model)
python trd_framework/run_phase1_complete.py \
    --mode general \
    --teacher-model qwen2.5:32b

# Train student model
python trd_framework/run_phase2_training.py \
    --dataset trd_framework/general_trd_complete_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-general-reasoning
```

## Performance Expectations

### Physics Mode

| Metric | Expected Value |
|--------|----------------|
| Physics Reasoning Score | 3.8-4.2 / 5.0 |
| Self-Correction Rate | 60-70% |
| Conservation Law Accuracy | >95% |
| Ravan Integration | Seamless |

### General Mode

| Metric | Expected Value |
|--------|----------------|
| General Reasoning Score | 3.5-4.0 / 5.0 |
| Self-Correction Rate | 55-65% |
| Domain Coverage | 8 domains |
| Versatility | High |

## Combining Both Modes

You can train **two separate models**:

1. **Physics Expert**: For Ravan system (physics mode)
2. **General Assistant**: For general tasks (general mode)

Or create a **hybrid model** by combining datasets:

```bash
# Merge datasets
cat trd_framework/ravan_trd_complete_dataset.jsonl \
    trd_framework/general_trd_complete_dataset.jsonl \
    > trd_framework/hybrid_trd_dataset.jsonl

# Train hybrid model
python trd_framework/run_phase2_training.py \
    --dataset trd_framework/hybrid_trd_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-hybrid
```

This creates a model that:
- Excels at quantum physics (from physics queries)
- Handles general reasoning (from general queries)
- Maintains small size (~3.2GB)

## Recommendations

### For Ravan Users

**Use Physics Mode** if you:
- Need quantum physics expertise
- Want to explain Ravan simulations
- Require physics-grounded reasoning
- Plan to integrate with Ravan system

### For General Users

**Use General Mode** if you:
- Need a versatile AI assistant
- Work across multiple domains
- Want broad reasoning capabilities
- Don't need physics specialization

### For Advanced Users

**Use Hybrid Mode** if you:
- Need both physics and general reasoning
- Have sufficient training time (~4-6 hours)
- Want a single model for everything
- Can afford slightly reduced specialization

## Conclusion

The TRD framework is **domain-agnostic by design**. The same core principles (chain-of-thought reasoning, self-correction, stabilizer loss) work across any domain.

**Key Insight**: TRD teaches "how to think", not "what to know". The domain knowledge comes from the queries and teacher responses, while the reasoning structure comes from the TRD framework itself.

This makes TRD incredibly flexible—you can create specialized models for:
- Medical diagnosis
- Legal reasoning
- Financial analysis
- Creative writing
- Software engineering
- Or any other domain!

Just generate domain-specific queries, run them through the TRD pipeline, and you'll get a small, efficient model with strong reasoning capabilities in that domain.

---

**Files Created**:
- `trd_framework/general_trd_integration.py` - General-purpose query generation
- `trd_framework/general_purpose_queries.jsonl` - 35 multi-domain queries
- `trd_framework/run_phase1_complete.py` - Updated to support both modes

**Next Steps**:
1. Choose your mode (physics or general)
2. Run Phase 1 with teacher model
3. Train student model with Phase 2
4. Evaluate with Phase 3
5. Deploy your specialized reasoning model!
