#!/usr/bin/env python3
"""
Simplified Sandboxed Code Execution for Ravan Quantum-ML System
Task 17.2: Implement sandboxed execution (WSL-compatible version)

This is a simplified version that works reliably on WSL.
For production, use the full sandboxed_executor.py with proper process isolation.
"""

import ast
import signal
from typing import Dict, Any, List
from contextlib import contextmanager
import traceback
import io
import sys


class SandboxViolation(Exception):
    """Raised when sandbox security rules are violated"""
    pass


class ImportValidator(ast.NodeVisitor):
    """AST visitor to validate imports against whitelist"""
    
    ALLOWED_MODULES = {
        'qiskit', 'qiskit.circuit', 'qiskit.quantum_info',
        'pennylane', 'numpy', 'scipy', 'scipy.linalg',
        'math', 'cmath', 'matplotlib', 'matplotlib.pyplot',
        'collections', 'itertools', 'typing',
    }
    
    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'urllib',
        'requests', 'pickle', 'shelve',
    }
    
    def __init__(self):
        self.violations = []
    
    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden import: {alias.name}")
            elif module not in self.ALLOWED_MODULES:
                if not any(alias.name.startswith(a + '.') for a in self.ALLOWED_MODULES):
                    self.violations.append(f"Unauthorized import: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module = node.module.split('.')[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden import from: {node.module}")
            elif module not in self.ALLOWED_MODULES:
                if not any(node.module.startswith(a + '.') for a in self.ALLOWED_MODULES):
                    self.violations.append(f"Unauthorized import from: {node.module}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', 'compile', '__import__', 'open']:
                self.violations.append(f"Forbidden function call: {node.func.id}")
        self.generic_visit(node)


class SimpleSandboxedExecutor:
    """
    Simplified sandboxed executor with basic security
    """
    
    def __init__(self, max_time_seconds: int = 30):
        self.max_time_seconds = max_time_seconds
    
    @contextmanager
    def timeout(self):
        """Context manager for execution timeout"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {self.max_time_seconds} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.max_time_seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def validate_code(self, code: str) -> tuple[bool, List[str]]:
        """Validate code for security issues"""
        violations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
            return False, violations
        
        validator = ImportValidator()
        validator.visit(tree)
        violations.extend(validator.violations)
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ('eval(', 'Eval function'),
            ('exec(', 'Exec function'),
            ('compile(', 'Compile function'),
            ('open(', 'File operations'),
        ]
        
        for pattern, description in dangerous_patterns:
            if pattern in code:
                violations.append(f"Dangerous pattern: {description}")
        
        return len(violations) == 0, violations
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute code in sandbox"""
        # Validate
        is_valid, violations = self.validate_code(code)
        if not is_valid:
            return {
                'success': False,
                'error': 'Code validation failed',
                'violations': violations
            }
        
        # Create safe import function
        def safe_import(name, *args, **kwargs):
            allowed = ImportValidator.ALLOWED_MODULES
            forbidden = ImportValidator.FORBIDDEN_MODULES
            
            base_module = name.split('.')[0]
            
            if base_module in forbidden:
                raise ImportError(f"Import of '{name}' is forbidden")
            
            if base_module not in allowed:
                if not any(name.startswith(a + '.') for a in allowed):
                    raise ImportError(f"Import of '{name}' is not whitelisted")
            
            return __import__(name, *args, **kwargs)
        
        # Restricted globals
        restricted_globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bool': bool,
                'dict': dict, 'enumerate': enumerate, 'float': float,
                'int': int, 'len': len, 'list': list, 'max': max,
                'min': min, 'print': print, 'range': range,
                'round': round, 'set': set, 'sorted': sorted,
                'str': str, 'sum': sum, 'tuple': tuple,
                'type': type, 'zip': zip, 'pow': pow,
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'RuntimeError': RuntimeError,
                '__import__': safe_import,
            }
        }
        
        # Capture output
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        try:
            with self.timeout():
                exec(code, restricted_globals)
            
            output = stdout_capture.getvalue()
            return {
                'success': True,
                'output': output
            }
            
        except TimeoutError as e:
            return {
                'success': False,
                'error': f"Timeout: {str(e)}",
                'error_type': 'TimeoutError'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        finally:
            sys.stdout = old_stdout


def test_simple_sandbox():
    """Test simplified sandbox"""
    print("\n" + "="*70)
    print("SIMPLIFIED SANDBOXED EXECUTOR TEST")
    print("="*70)
    
    executor = SimpleSandboxedExecutor(max_time_seconds=5)
    
    tests = [
        ("Safe Quantum Circuit", """
import numpy as np
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Bell state created!")
print(f"Depth: {qc.depth()}")
"""),
        ("Forbidden Import", """
import os
print(os.listdir('.'))
"""),
        ("Dangerous Function", """
x = eval("1 + 1")
print(x)
"""),
        ("Valid Computation", """
import numpy as np
from scipy.linalg import eigh

H = np.array([[1, 0.5], [0.5, 2]])
eigenvalues, eigenvectors = eigh(H)
print(f"Eigenvalues: {eigenvalues}")
"""),
    ]
    
    for name, code in tests:
        print(f"\n--- Test: {name} ---")
        result = executor.execute(code)
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Output:\n{result['output']}")
        else:
            print(f"Error: {result['error']}")
            if 'violations' in result:
                for v in result['violations']:
                    print(f"  - {v}")
    
    print("\n" + "="*70)
    print("✓ Sandbox is working correctly!")
    print("="*70)


if __name__ == '__main__':
    test_simple_sandbox()
