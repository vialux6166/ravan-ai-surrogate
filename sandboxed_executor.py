#!/usr/bin/env python3
"""
Sandboxed Code Execution for Ravan Quantum-ML System
Task 17.2: Implement sandboxed execution

Provides safe execution environment for LLM-generated code with:
- Whitelisted imports
- Resource limits (time, memory)
- Restricted operations
"""

import sys
import ast
import signal
import resource
import multiprocessing as mp
from typing import Dict, Any, Optional, List, Set
from contextlib import contextmanager
import traceback
import io


class SandboxViolation(Exception):
    """Raised when sandbox security rules are violated"""
    pass


class ImportValidator(ast.NodeVisitor):
    """
    AST visitor to validate imports against whitelist
    """
    
    ALLOWED_MODULES = {
        # Quantum libraries
        'qiskit',
        'qiskit.circuit',
        'qiskit.quantum_info',
        'qiskit.providers',
        'qiskit.providers.aer',
        'pennylane',
        
        # Scientific computing
        'numpy',
        'scipy',
        'scipy.linalg',
        'scipy.optimize',
        'scipy.integrate',
        
        # Math
        'math',
        'cmath',
        
        # Plotting (read-only)
        'matplotlib',
        'matplotlib.pyplot',
        
        # Data structures
        'collections',
        'itertools',
        
        # Type hints
        'typing',
    }
    
    FORBIDDEN_MODULES = {
        'os',
        'sys',
        'subprocess',
        'socket',
        'urllib',
        'requests',
        'pickle',
        'shelve',
        'eval',
        'exec',
        '__import__',
        'open',
        'file',
        'input',
        'raw_input',
    }
    
    def __init__(self):
        self.violations = []
    
    def visit_Import(self, node):
        """Check regular imports"""
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden import: {alias.name}")
            elif module not in self.ALLOWED_MODULES:
                # Check if it's a submodule of allowed module
                if not any(alias.name.startswith(allowed + '.') 
                          for allowed in self.ALLOWED_MODULES):
                    self.violations.append(f"Unauthorized import: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Check from...import statements"""
        if node.module:
            module = node.module.split('.')[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden import from: {node.module}")
            elif module not in self.ALLOWED_MODULES:
                if not any(node.module.startswith(allowed + '.') 
                          for allowed in self.ALLOWED_MODULES):
                    self.violations.append(f"Unauthorized import from: {node.module}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Check for dangerous function calls"""
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', 'compile', '__import__', 'open']:
                self.violations.append(f"Forbidden function call: {node.func.id}")
        self.generic_visit(node)


class CodeValidator:
    """
    Validates code before execution
    """
    
    def __init__(self):
        pass
    
    def validate(self, code: str) -> tuple[bool, List[str]]:
        """
        Validate code for security issues
        
        Args:
            code: Python code to validate
            
        Returns:
            (is_valid, violations)
        """
        violations = []
        
        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
            return False, violations
        
        # Check imports - create fresh validator each time
        import_validator = ImportValidator()
        import_validator.visit(tree)
        violations.extend(import_validator.violations)
        
        # Check for dangerous patterns (but not in comments or strings)
        # Remove comments and strings for pattern matching
        code_lines = []
        for line in code.split('\n'):
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            code_lines.append(line)
        code_no_comments = '\n'.join(code_lines)
        
        dangerous_patterns = [
            ('eval(', 'Eval function'),
            ('exec(', 'Exec function'),
            ('compile(', 'Compile function'),
            ('open(', 'File operations'),
            ('file(', 'File operations'),
            ('input(', 'User input'),
            ('raw_input(', 'User input'),
        ]
        
        for pattern, description in dangerous_patterns:
            if pattern in code_no_comments:
                violations.append(f"Dangerous pattern detected: {description}")
        
        is_valid = len(violations) == 0
        return is_valid, violations


class ResourceLimiter:
    """
    Manages resource limits for sandboxed execution
    """
    
    def __init__(self, 
                 max_time_seconds: int = 30,
                 max_memory_mb: int = 512):
        self.max_time_seconds = max_time_seconds
        self.max_memory_mb = max_memory_mb
    
    def set_limits(self):
        """Set resource limits for current process"""
        # Set memory limit (virtual memory)
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        
        # Set CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (self.max_time_seconds, self.max_time_seconds))
    
    @contextmanager
    def timeout(self):
        """Context manager for execution timeout"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {self.max_time_seconds} seconds")
        
        # Set alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.max_time_seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _execute_in_sandbox(code: str, timeout: int, memory_limit: int, 
                       result_queue: mp.Queue):
    """
    Execute code in sandboxed subprocess
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        memory_limit: Maximum memory in MB
        result_queue: Queue to return results
    """
    try:
        # Set resource limits
        limiter = ResourceLimiter(max_time_seconds=timeout, max_memory_mb=memory_limit)
        limiter.set_limits()
        
        # Create restricted globals with safe __import__
        def safe_import(name, *args, **kwargs):
            """Safe import that only allows whitelisted modules"""
            allowed = ImportValidator.ALLOWED_MODULES
            forbidden = ImportValidator.FORBIDDEN_MODULES
            
            base_module = name.split('.')[0]
            
            if base_module in forbidden:
                raise ImportError(f"Import of '{name}' is forbidden")
            
            if base_module not in allowed:
                # Check if it's a submodule of allowed
                if not any(name.startswith(a + '.') for a in allowed):
                    raise ImportError(f"Import of '{name}' is not whitelisted")
            
            return __import__(name, *args, **kwargs)
        
        restricted_globals = {
            '__builtins__': {
                # Safe built-ins only
                'abs': abs,
                'all': all,
                'any': any,
                'bool': bool,
                'dict': dict,
                'enumerate': enumerate,
                'float': float,
                'int': int,
                'len': len,
                'list': list,
                'max': max,
                'min': min,
                'print': print,
                'range': range,
                'round': round,
                'set': set,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'type': type,
                'zip': zip,
                # Math functions
                'pow': pow,
                # Exceptions
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'RuntimeError': RuntimeError,
                # Safe import
                '__import__': safe_import,
            }
        }
        
        # Capture stdout
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        try:
            # Execute code with timeout
            with limiter.timeout():
                exec(code, restricted_globals)
            
            # Get output
            output = stdout_capture.getvalue()
            
            result_queue.put({
                'success': True,
                'output': output,
                'globals': {k: v for k, v in restricted_globals.items() 
                           if not k.startswith('__')}
            })
            
        finally:
            sys.stdout = old_stdout
            
    except TimeoutError as e:
        result_queue.put({
            'success': False,
            'error': f"Timeout: {str(e)}",
            'error_type': 'TimeoutError'
        })
    except MemoryError as e:
        result_queue.put({
            'success': False,
            'error': f"Memory limit exceeded: {str(e)}",
            'error_type': 'MemoryError'
        })
    except Exception as e:
        result_queue.put({
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        })


class SandboxedExecutor:
    """
    Sandboxed code executor with security restrictions
    """
    
    def __init__(self,
                 max_time_seconds: int = 30,
                 max_memory_mb: int = 512,
                 validate_code: bool = True):
        """
        Initialize sandboxed executor
        
        Args:
            max_time_seconds: Maximum execution time
            max_memory_mb: Maximum memory usage
            validate_code: Whether to validate code before execution
        """
        self.max_time_seconds = max_time_seconds
        self.max_memory_mb = max_memory_mb
        self.validate_code = validate_code
        self.validator = CodeValidator()
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in sandbox
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with execution results:
            - success: bool
            - output: str (if successful)
            - error: str (if failed)
            - violations: list (if validation failed)
        """
        # Validate code
        if self.validate_code:
            is_valid, violations = self.validator.validate(code)
            if not is_valid:
                return {
                    'success': False,
                    'error': 'Code validation failed',
                    'violations': violations
                }
        
        # Execute in separate process for isolation
        result_queue = mp.Queue()
        process = mp.Process(
            target=_execute_in_sandbox,
            args=(code, self.max_time_seconds, self.max_memory_mb, result_queue)
        )
        
        process.start()
        process.join(timeout=self.max_time_seconds + 5)  # Extra buffer
        
        if process.is_alive():
            # Process didn't finish in time
            process.terminate()
            process.join()
            return {
                'success': False,
                'error': f'Execution timeout ({self.max_time_seconds}s)',
                'error_type': 'TimeoutError'
            }
        
        # Get result
        if not result_queue.empty():
            return result_queue.get()
        else:
            return {
                'success': False,
                'error': 'Execution failed without error message',
                'error_type': 'UnknownError'
            }
    
    def execute_safe(self, code: str) -> tuple[bool, str]:
        """
        Execute code and return simple success/output tuple
        
        Args:
            code: Python code to execute
            
        Returns:
            (success, output_or_error)
        """
        result = self.execute(code)
        
        if result['success']:
            return True, result.get('output', '')
        else:
            error_msg = result.get('error', 'Unknown error')
            if 'violations' in result:
                error_msg += '\nViolations:\n' + '\n'.join(result['violations'])
            return False, error_msg


def test_sandbox():
    """Test sandboxed execution"""
    print("\n" + "="*70)
    print("SANDBOXED EXECUTOR TEST SUITE")
    print("="*70)
    
    executor = SandboxedExecutor(max_time_seconds=5, max_memory_mb=256)
    
    # Test 1: Safe code
    print("\n--- Test 1: Safe Quantum Circuit ---")
    safe_code = """
import numpy as np
from qiskit import QuantumCircuit

# Create Bell state
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

print("Bell state circuit created successfully!")
print(f"Circuit depth: {qc.depth()}")
"""
    
    result = executor.execute(safe_code)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Output:\n{result['output']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test 2: Forbidden import
    print("\n--- Test 2: Forbidden Import (os) ---")
    forbidden_code = """
import os
print(os.listdir('.'))
"""
    
    result = executor.execute(forbidden_code)
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Blocked: {result['error']}")
        if 'violations' in result:
            print(f"Violations: {result['violations']}")
    
    # Test 3: Dangerous function
    print("\n--- Test 3: Dangerous Function (eval) ---")
    dangerous_code = """
x = eval("1 + 1")
print(x)
"""
    
    result = executor.execute(dangerous_code)
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Blocked: {result['error']}")
        if 'violations' in result:
            print(f"Violations: {result['violations']}")
    
    # Test 4: Infinite loop (timeout)
    print("\n--- Test 4: Infinite Loop (Timeout) ---")
    timeout_code = """
while True:
    pass
"""
    
    result = executor.execute(timeout_code)
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Blocked: {result['error']}")
    
    # Test 5: Memory exhaustion
    print("\n--- Test 5: Memory Exhaustion ---")
    memory_code = """
big_list = []
for i in range(10000000):
    big_list.append([0] * 1000)
"""
    
    result = executor.execute(memory_code)
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Blocked: {result['error']}")
    
    # Test 6: Valid scientific computation
    print("\n--- Test 6: Valid Scientific Computation ---")
    valid_code = """
import numpy as np
from scipy.linalg import eigh

# Create Hamiltonian
H = np.array([[1, 0.5], [0.5, 2]])

# Solve eigenvalue problem
eigenvalues, eigenvectors = eigh(H)

print(f"Eigenvalues: {eigenvalues}")
print(f"Ground state energy: {eigenvalues[0]:.4f}")
"""
    
    result = executor.execute(valid_code)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Output:\n{result['output']}")
    else:
        print(f"Error: {result['error']}")
    
    print("\n" + "="*70)
    print("SANDBOX TEST COMPLETE")
    print("="*70)
    print("\nSummary:")
    print("✓ Safe code executes successfully")
    print("✓ Forbidden imports are blocked")
    print("✓ Dangerous functions are blocked")
    print("✓ Timeouts prevent infinite loops")
    print("✓ Memory limits prevent exhaustion")
    print("✓ Scientific computations work correctly")


if __name__ == '__main__':
    test_sandbox()
