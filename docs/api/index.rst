API Reference
=============

This section contains the complete API reference for the Ravan Quantum-ML System.

.. toctree::
   :maxdepth: 2

   core
   simulators
   models
   llm
   utils

Core API
--------

.. automodule:: ravan
   :members:
   :undoc-members:
   :show-inheritance:

Main Classes
~~~~~~~~~~~~

Ravan
^^^^^

.. autoclass:: ravan.Ravan
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ravan.simulate
.. autofunction:: ravan.train
.. autofunction:: ravan.predict
.. autofunction:: ravan.ask
