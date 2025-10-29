Core API
========

Main API interface for the Ravan system.

Ravan Class
-----------

The main entry point for the Ravan Quantum-ML System.

.. autoclass:: ravan.Ravan
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__

   .. automethod:: __init__
   .. automethod:: simulate
   .. automethod:: train
   .. automethod:: predict
   .. automethod:: explain
   .. automethod:: generate_code
   .. automethod:: query

Convenience Functions
---------------------

Quick access functions for common operations.

simulate
~~~~~~~~

.. autofunction:: ravan.simulate

train
~~~~~

.. autofunction:: ravan.train

predict
~~~~~~~

.. autofunction:: ravan.predict

ask
~~~

.. autofunction:: ravan.ask
