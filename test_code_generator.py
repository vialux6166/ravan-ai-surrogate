#!/usr/bin/env python3
"""
Test Code Generator
Task 17.1: Test code generation with syntax validation
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import ast
from src.llm.code_generator import CodeGenerator, validate_code


def test_code_validation():
    """Test code validation"""
    print("\n" + "="*70)
    print("TEST 1: Code Validation")
    print("="*70)
    
    # Mock LLM adapter
    class MockLLM:
        def query(self, prompt, **kwargs):
            return "print('hello')"
    
    generator = CodeGenerator(MockLLM())
    
    # Test valid code
    valid_code = """
from qiskit import QuantumCircuit
import numpy as np

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
    
    is_valid, error = generator.validate_code(valid_code)
    assert is_valid, f"Valid code rejected: {error}"
    print("✓ Valid code accepted")
    
    # Test invalid syntax
    invalid_syntax = "def foo(\nprint('bad')"
    is_valid, error = generator.validate_code(invalid_syntax)
    assert not is_valid, "Invalid syntax not detected"
    print("✓ Invalid syntax detected")
    
    # Test blocked import
    blocked_import = "import os\nos.system('rm -rf /')"
    is_valid, error = generator.validate_code(blocked_import)
    assert not is_valid, "Blocked import not detected"
    print(f"✓ Blocked import detected: {error}")
    
    # Test eval/exec
    dangerous_code = "eval('malicious code')"
    is_valid, error = generator.validate_code(dangerous_code)
    assert not is_valid, "Dangerous eval not detected"
    print(f"✓ Dangerous eval detected: {error}")
    
    return True


def test_code_extraction():
    """Test code extraction from LLM response"""
    print("\n" + "="*70)
    print("TEST 2: Code Extraction")
    print("="*70)
    
    class MockLLM:
        def query(self, prompt, **kwargs):
            return "print('hello')"
    
    generator = CodeGenerator(MockLLM())
    
    # Test markdown code block
    response1 = """Here's the code:

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
```

That's it!"""
    
    code1 = generator._extract_code(response1)
    assert "QuantumCircuit" in code1
    assert "Here's" not in code1
    print("✓ Extracted code from markdown block")
    
    # Test generic code block
    response2 = """```
import numpy as np
x = np.array([1, 2, 3])
```"""
    
    code2 = generator._extract_code(response2)
    assert "numpy" in code2
    print("✓ Extracted code from generic block")
    
    # Test plain code
    response3 = "print('hello world')"
    code3 = generator._extract_code(response3)
    assert code3 == "print('hello world')"
    print("✓ Extracted plain code")
    
    return True


def test_import_validation():
    """Test import validation"""
    print("\n" + "="*70)
    print("TEST 3: Import Validation")
    print("="*70)
    
    class MockLLM:
        def query(self, prompt, **kwargs):
            return "print('hello')"
    
    generator = CodeGenerator(MockLLM())
    
    # Test allowed imports
    allowed_imports = [
        "import qiskit",
        "from qiskit import QuantumCircuit",
        "import numpy as np",
        "from scipy import optimize",
        "import matplotlib.pyplot as plt",
    ]
    
    for imp in allowed_imports:
        is_valid, error = generator.validate_code(imp)
        assert is_valid, f"Allowed import rejected: {imp}"
    
    print(f"✓ All {len(allowed_imports)} allowed imports accepted")
    
    # Test blocked imports
    blocked_imports = [
        "import os",
        "import subprocess",
        "import sys",
        "from os import system",
    ]
    
    for imp in blocked_imports:
        is_valid, error = generator.validate_code(imp)
        assert not is_valid, f"Blocked import not detected: {imp}"
    
    print(f"✓ All {len(blocked_imports)} blocked imports rejected")
    
    return True


def test_code_formatting():
    """Test code formatting"""
    print("\n" + "="*70)
    print("TEST 4: Code Formatting")
    print("="*70)
    
    class MockLLM:
        def query(self, prompt, **kwargs):
            return "print('hello')"
    
    generator = CodeGenerator(MockLLM())
    
    # Test formatting
    messy_code = """
import numpy as np   


def foo():   
    return 1   


x = foo()   
"""
    
    formatted = generator.format_code(messy_code)
    
    # Check trailing whitespace removed
    for line in formatted.split('\n'):
        assert line == line.rstrip(), "Trailing whitespace not removed"
    
    # Check excessive blank lines reduced
    assert '\n\n\n' not in formatted, "Excessive blank lines not removed"
    
    print("✓ Code formatted correctly")
    
    return True


def test_documentation_addition():
    """Test adding documentation"""
    print("\n" + "="*70)
    print("TEST 5: Documentation Addition")
    print("="*70)
    
    class MockLLM:
        def query(self, prompt, **kwargs):
            return "print('hello')"
    
    generator = CodeGenerator(MockLLM())
    
    code = "print('hello')"
    description = "Simple hello world program"
    
    documented = generator.add_documentation(code, description)
    
    assert '"""' in documented, "Docstring not added"
    assert description in documented, "Description not in docstring"
    assert "Ravan" in documented, "Attribution not added"
    assert code in documented, "Original code not preserved"
    
    print("✓ Documentation added correctly")
    
    return True


def test_syntax_validation():
    """Test AST-based syntax validation"""
    print("\n" + "="*70)
    print("TEST 6: Syntax Validation")
    print("="*70)
    
    class MockLLM:
        def query(self, prompt, **kwargs):
            return "print('hello')"
    
    generator = CodeGenerator(MockLLM())
    
    # Valid Python
    valid_codes = [
        "x = 1",
        "def foo(): return 1",
        "class Bar: pass",
        "[x for x in range(10)]",
    ]
    
    for code in valid_codes:
        is_valid, error = generator.validate_code(code)
        assert is_valid, f"Valid code rejected: {code}"
    
    print(f"✓ {len(valid_codes)} valid code samples accepted")
    
    # Invalid Python
    invalid_codes = [
        "def foo(",  # Incomplete
        "x = ",  # Incomplete
        "if True",  # Missing colon
        "for x in",  # Incomplete
    ]
    
    for code in invalid_codes:
        is_valid, error = generator.validate_code(code)
        assert not is_valid, f"Invalid code not detected: {code}"
    
    print(f"✓ {len(invalid_codes)} invalid code samples rejected")
    
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("CODE GENERATOR TEST SUITE")
    print("="*70)
    print("\nNote: Testing validation logic without actual LLM")
    
    results = {}
    
    # Run tests
    results['validation'] = test_code_validation()
    results['extraction'] = test_code_extraction()
    results['imports'] = test_import_validation()
    results['formatting'] = test_code_formatting()
    results['documentation'] = test_documentation_addition()
    results['syntax'] = test_syntax_validation()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nCode Generator is working correctly!")
        print("- Syntax validation prevents invalid code")
        print("- Import filtering blocks dangerous operations")
        print("- Code extraction handles various formats")
        print("- Formatting improves readability")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
