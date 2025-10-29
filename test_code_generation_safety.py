#!/usr/bin/env python3
"""
Code Generation Safety Tests for Ravan Quantum-ML System
Task 17.3: Test code generation safety

Tests the sandboxed executor with various malicious inputs
to ensure security measures are effective.
"""

import sys
sys.path.insert(0, '.')

from sandboxed_executor_simple import SimpleSandboxedExecutor


class SecurityTester:
    """
    Tests sandbox security with malicious code attempts
    """
    
    def __init__(self):
        self.executor = SimpleSandboxedExecutor(max_time_seconds=5)
        self.test_results = []
    
    def run_test(self, test_name: str, code: str, should_block: bool = True):
        """
        Run a security test
        
        Args:
            test_name: Name of the test
            code: Code to test
            should_block: Whether the code should be blocked
        """
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")
        print(f"Expected: {'BLOCKED' if should_block else 'ALLOWED'}")
        
        result = self.executor.execute(code)
        success = result['success']
        
        # Check if result matches expectation
        test_passed = (not success) == should_block
        
        print(f"Result: {'BLOCKED' if not success else 'ALLOWED'}")
        print(f"Status: {'✓ PASS' if test_passed else '✗ FAIL'}")
        
        if not success:
            print(f"Reason: {result.get('error', 'Unknown')}")
            if 'violations' in result:
                for v in result['violations']:
                    print(f"  - {v}")
        else:
            print(f"Output: {result.get('output', '(no output)')[:100]}")
        
        self.test_results.append({
            'name': test_name,
            'passed': test_passed,
            'blocked': not success
        })
        
        return test_passed
    
    def run_all_tests(self):
        """Run comprehensive security test suite"""
        print("\n" + "="*70)
        print("CODE GENERATION SAFETY TEST SUITE")
        print("="*70)
        print("\nTesting sandbox security against malicious code attempts...")
        
        # Test 1: File system access
        self.run_test(
            "File System Access (os.listdir)",
            """
import os
files = os.listdir('.')
print(files)
""",
            should_block=True
        )
        
        # Test 2: File operations
        self.run_test(
            "File Operations (open)",
            """
with open('/etc/passwd', 'r') as f:
    print(f.read())
""",
            should_block=True
        )
        
        # Test 3: Subprocess execution
        self.run_test(
            "Subprocess Execution",
            """
import subprocess
result = subprocess.run(['ls', '-la'], capture_output=True)
print(result.stdout)
""",
            should_block=True
        )
        
        # Test 4: Network access
        self.run_test(
            "Network Access (socket)",
            """
import socket
s = socket.socket()
s.connect(('google.com', 80))
""",
            should_block=True
        )
        
        # Test 5: HTTP requests
        self.run_test(
            "HTTP Requests",
            """
import requests
r = requests.get('https://google.com')
print(r.text)
""",
            should_block=True
        )
        
        # Test 6: Eval function
        self.run_test(
            "Eval Function",
            """
code = "print('malicious')"
eval(code)
""",
            should_block=True
        )
        
        # Test 7: Exec function
        self.run_test(
            "Exec Function",
            """
code = "import os; os.system('ls')"
exec(code)
""",
            should_block=True
        )
        
        # Test 8: Compile function
        self.run_test(
            "Compile Function",
            """
code = compile("print('test')", '<string>', 'exec')
exec(code)
""",
            should_block=True
        )
        
        # Test 9: Dynamic import
        self.run_test(
            "Dynamic Import (__import__)",
            """
os_module = __import__('os')
print(os_module.listdir('.'))
""",
            should_block=True
        )
        
        # Test 10: Pickle deserialization
        self.run_test(
            "Pickle Deserialization",
            """
import pickle
data = pickle.loads(b'malicious_data')
""",
            should_block=True
        )
        
        # Test 11: Infinite loop (timeout)
        self.run_test(
            "Infinite Loop (Timeout Test)",
            """
while True:
    pass
""",
            should_block=True
        )
        
        # Test 12: Excessive recursion
        self.run_test(
            "Excessive Recursion",
            """
def recurse():
    recurse()
recurse()
""",
            should_block=True
        )
        
        # Test 13: Safe quantum circuit (should ALLOW)
        self.run_test(
            "Safe Quantum Circuit",
            """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

print(f"Circuit created with {qc.num_qubits} qubits")
print(f"Circuit depth: {qc.depth()}")
""",
            should_block=False
        )
        
        # Test 14: Safe numpy computation (should ALLOW)
        self.run_test(
            "Safe NumPy Computation",
            """
import numpy as np

# Create random matrix
A = np.random.randn(3, 3)
eigenvalues = np.linalg.eigvals(A)

print(f"Matrix shape: {A.shape}")
print(f"Eigenvalues: {eigenvalues}")
""",
            should_block=False
        )
        
        # Test 15: Safe scipy computation (should ALLOW)
        self.run_test(
            "Safe SciPy Computation",
            """
import numpy as np
from scipy.linalg import eigh

# Hamiltonian matrix
H = np.array([[1.0, 0.5], [0.5, 2.0]])

# Solve eigenvalue problem
eigenvalues, eigenvectors = eigh(H)

print(f"Ground state energy: {eigenvalues[0]:.4f}")
print(f"Excited state energy: {eigenvalues[1]:.4f}")
""",
            should_block=False
        )
        
        # Test 16: Safe mathematical computation (should ALLOW)
        self.run_test(
            "Safe Mathematical Computation",
            """
import math
import numpy as np

# Calculate quantum tunneling probability
def tunneling_probability(E, V0, width):
    k = math.sqrt(2 * (V0 - E))
    return math.exp(-2 * k * width)

E = 3.0
V0 = 5.0
width = 1.5

T = tunneling_probability(E, V0, width)
print(f"Tunneling probability: {T:.6f}")
""",
            should_block=False
        )
        
        # Generate summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = total - passed
        
        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Success rate: {100 * passed / total:.1f}%")
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.test_results:
                if not r['passed']:
                    print(f"  ✗ {r['name']}")
        
        print("\n" + "="*70)
        print("SECURITY VALIDATION")
        print("="*70)
        
        # Check critical security tests
        critical_tests = [
            "File System Access (os.listdir)",
            "File Operations (open)",
            "Subprocess Execution",
            "Network Access (socket)",
            "Eval Function",
            "Exec Function",
        ]
        
        critical_passed = all(
            r['passed'] and r['blocked']
            for r in self.test_results
            if r['name'] in critical_tests
        )
        
        if critical_passed:
            print("\n✅ ALL CRITICAL SECURITY TESTS PASSED")
            print("\nThe sandbox successfully blocks:")
            print("  ✓ File system access")
            print("  ✓ Network operations")
            print("  ✓ Subprocess execution")
            print("  ✓ Code injection (eval/exec)")
            print("  ✓ Dangerous imports")
            print("\nThe sandbox allows:")
            print("  ✓ Quantum circuit creation")
            print("  ✓ Scientific computations")
            print("  ✓ Mathematical operations")
        else:
            print("\n⚠️  CRITICAL SECURITY TESTS FAILED")
            print("\nThe sandbox has security vulnerabilities!")
        
        print("\n" + "="*70)
        
        return critical_passed


def main():
    """Run security test suite"""
    tester = SecurityTester()
    tester.run_all_tests()


if __name__ == '__main__':
    main()
