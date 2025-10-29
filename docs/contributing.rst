Contributing
============

We welcome contributions to the Ravan Quantum-ML System!

How to Contribute
-----------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass
6. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   # Clone your fork
   git clone <your-fork-url>
   cd ravan-quantum-ml

   # Create development environment
   python3 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-llm.txt

   # Install development tools
   pip install pytest pytest-cov black flake8 mypy

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=. --cov-report=html

   # Run specific test file
   pytest tests/test_code_sandbox.py -v

Code Style
----------

We follow PEP 8 style guidelines:

.. code-block:: bash

   # Format code
   black .

   # Check style
   flake8 .

   # Type checking
   mypy .

Adding Documentation
--------------------

Documentation is built with Sphinx:

.. code-block:: bash

   # Build documentation
   cd docs
   make html

   # View documentation
   open _build/html/index.html

Guidelines
----------

* Write clear, descriptive commit messages
* Add docstrings to all public functions/classes
* Include type hints where appropriate
* Write tests for new features
* Update documentation for API changes
* Follow existing code style

Questions?
----------

Feel free to open an issue for questions or discussions.
