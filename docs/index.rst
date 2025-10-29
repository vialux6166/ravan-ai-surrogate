Ravan Quantum-ML System Documentation
======================================

Welcome to the Ravan Quantum-ML System documentation. Ravan is a production-ready quantum simulation and machine learning system with optional AI assistance powered by large language models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   user_guide
   examples
   contributing

Overview
--------

Ravan combines three quantum simulators with machine learning to predict quantum behaviors in milliseconds instead of hours:

1. **Quantum Circuit Simulator** - Entanglement and quantum gates
2. **Schrödinger Solver** - Quantum tunneling through barriers
3. **Harmonic Oscillator** - Energy quantization and coherent states

Features
--------

Core Features
~~~~~~~~~~~~~

* **GPU-Accelerated** - CUDA support for PyTorch and XGBoost
* **VRAM Management** - Intelligent memory management for RTX 3090
* **Parallel Execution** - Multi-core CPU parallelization
* **Physics Validation** - Automatic constraint checking
* **Uncertainty Quantification** - MC Dropout for confidence intervals

LLM Integration (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Natural Language Interface** - Describe simulations in plain English
* **Results Explanation** - Get physics context automatically
* **Code Generation** - Generate Python/Qiskit code from descriptions
* **Interactive Chat** - Ask questions about quantum mechanics

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone <repository-url>
   cd ravan-quantum-ml

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install core dependencies
   pip install -r requirements.txt

   # Optional: Install LLM dependencies (requires GPU)
   pip install -r requirements-llm.txt

Basic Usage
~~~~~~~~~~~

Without LLM:

.. code-block:: python

   from ravan import Ravan

   # Initialize
   ravan = Ravan()

   # Run simulation
   results = ravan.simulate({
       'simulator': 'schrodinger',
       'V0': 5.0,
       'k0': 4.0,
       'barrier_width': 1.5
   })

   print(results)

With LLM:

.. code-block:: python

   from ravan import Ravan

   # Initialize with LLM
   with Ravan(use_llm=True) as ravan:
       # Natural language simulation
       results = ravan.simulate(
           "quantum tunneling with high barrier",
           explain=True
       )
       
       print(results['explanation'])

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
