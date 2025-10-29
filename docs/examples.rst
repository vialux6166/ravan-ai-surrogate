Examples
========

Practical examples of using the Ravan Quantum-ML System.

Basic Examples
--------------

See ``examples/basic_usage.py`` for complete working examples.

Quantum Circuit Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ravan import Ravan

   ravan = Ravan()
   
   # Create Bell state
   results = ravan.simulate({
       'simulator': 'quantum_circuit',
       'n_qubits': 2,
       'shots': 1000,
       'gate_sequence': 'bell'
   })
   
   print(f"Entropy: {results['entropy']:.4f} bits")

Quantum Tunneling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High barrier tunneling
   results = ravan.simulate({
       'simulator': 'schrodinger',
       'V0': 8.0,
       'k0': 5.0,
       'barrier_width': 2.0
   })
   
   print(f"Transmission: {results['transmission']:.4f}")

Advanced Examples
-----------------

Natural Language Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with Ravan(use_llm=True) as ravan:
       results = ravan.simulate(
           "quantum tunneling with narrow wavepacket",
           explain=True
       )
       print(results['explanation'])

Model Training
~~~~~~~~~~~~~~

.. code-block:: python

   ravan = Ravan()
   
   model = ravan.train(
       dataset_path='data/qc_dataset.h5',
       model_type='mlp',
       output_path='models/mlp_model'
   )

Code Generation
~~~~~~~~~~~~~~~

.. code-block:: python

   with Ravan(use_llm=True) as ravan:
       code = ravan.generate_code(
           "Create a 3-qubit GHZ state",
           code_type='quantum_circuit'
       )
       print(code)

More Examples
-------------

See the ``examples/`` directory for more complete examples.
