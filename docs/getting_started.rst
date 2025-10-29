Getting Started
===============

This guide will help you get started with the Ravan Quantum-ML System.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.8 or higher
* NVIDIA GPU with 8GB+ VRAM (optional, for ML training)
* NVIDIA GPU with 20GB+ VRAM (optional, for LLM features)
* CUDA 12.x or later (for GPU features)

Core Installation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone <repository-url>
   cd ravan-quantum-ml

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install core dependencies
   pip install -r requirements.txt

Optional: LLM Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install LLM dependencies (requires GPU with 20GB+ VRAM)
   pip install -r requirements-llm.txt

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run tests
   pytest tests/ -v

   # Check GPU availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

First Steps
-----------

Basic Simulation
~~~~~~~~~~~~~~~~

Run your first quantum simulation:

.. code-block:: python

   from ravan import Ravan

   # Initialize Ravan
   ravan = Ravan()

   # Run Schrödinger simulation
   results = ravan.simulate({
       'simulator': 'schrodinger',
       'V0': 5.0,              # Barrier height (eV)
       'k0': 4.0,              # Initial momentum
       'barrier_width': 1.5    # Barrier width (nm)
   })

   # Display results
   print(f"Transmission: {results['transmission']:.4f}")
   print(f"Reflection: {results['reflection']:.4f}")

Try Different Simulators
~~~~~~~~~~~~~~~~~~~~~~~~

Quantum Circuit:

.. code-block:: python

   results = ravan.simulate({
       'simulator': 'quantum_circuit',
       'n_qubits': 2,
       'shots': 1000,
       'gate_sequence': 'bell'
   })

   print(f"Entropy: {results['entropy']:.4f} bits")

Harmonic Oscillator:

.. code-block:: python

   results = ravan.simulate({
       'simulator': 'harmonic',
       'oscillator_length': 1.0,
       'basis_size': 10,
       'initial_n': 2
   })

   print(f"Ground state energy: {results['energy_0']:.4f}")

Using LLM Features
~~~~~~~~~~~~~~~~~~

If you have LLM installed:

.. code-block:: python

   from ravan import Ravan

   # Initialize with LLM
   with Ravan(use_llm=True) as ravan:
       # Natural language simulation
       results = ravan.simulate(
           "quantum tunneling with high barrier",
           explain=True
       )
       
       # Get explanation
       print(results['explanation'])
       
       # Ask questions
       answer = ravan.query("What is quantum entanglement?")
       print(answer)
       
       # Generate code
       code = ravan.generate_code(
           "Create a Bell state circuit with measurement"
       )
       print(code)

Next Steps
----------

* Read the :doc:`user_guide` for detailed usage
* Explore :doc:`examples` for more use cases
* Check the :doc:`api/index` for complete API reference
* See :doc:`contributing` to contribute to the project

Troubleshooting
---------------

LLM Not Loading
~~~~~~~~~~~~~~~

**Problem**: LLM fails to load

**Solutions**:

* Check GPU VRAM (need 20GB+)
* Install dependencies: ``pip install -r requirements-llm.txt``
* Check CUDA version: ``nvidia-smi``

Slow Performance
~~~~~~~~~~~~~~~~

**Problem**: Simulations are slow

**Solutions**:

* Enable GPU: Check ``torch.cuda.is_available()``
* Reduce sample size for testing
* Disable LLM: Use ``use_llm=False``

Import Errors
~~~~~~~~~~~~~

**Problem**: Module not found errors

**Solutions**:

* Activate virtual environment
* Install requirements: ``pip install -r requirements.txt``
* Check Python version: ``python --version`` (need 3.8+)

Getting Help
------------

* **Documentation**: This documentation
* **Examples**: ``examples/`` directory
* **Tests**: Run ``pytest tests/`` to verify installation
* **Issues**: GitHub Issues (if applicable)
