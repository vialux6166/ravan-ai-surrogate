#!/usr/bin/env python3
"""
Simulation Performance Profiling
Task 20.1: Profile simulation performance

Uses cProfile and line_profiler to identify bottlenecks
"""

import sys
sys.path.insert(0, '.')

import cProfile
import pstats
import io
import time
import numpy as np
from pathlib import Path

from schrodinger_solver import SchrodingerSolver
from quantum_circuit_simulator import QuantumCircuitSimulator
from harmonic_oscillator import HarmonicOscillatorModule


def profile_schrodinger():
    """Profile Schrödinger solver"""
    print("\n" + "="*70)
    print("PROFILING: Schrödinger Solver")
    print("="*70)
    
    solver = SchrodingerSolver()
    
    # Test parameters
    params = {
        'V0': 5.0,
        'barrier_width': 1.5,
        'k0': 4.5,
        'sigma': 1.0,
        'x0': -5.0
    }
    
    # Profile execution
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple times for better statistics
    n_runs = 100
    start_time = time.time()
    
    for _ in range(n_runs):
        result = solver.run(params)
    
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    # Print statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(f"\nTotal runs: {n_runs}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per run: {elapsed/n_runs*1000:.2f}ms")
    print(f"Throughput: {n_runs/elapsed:.2f} simulations/second")
    
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())
    
    return elapsed / n_runs


def profile_quantum_circuit():
    """Profile quantum circuit simulator"""
    print("\n" + "="*70)
    print("PROFILING: Quantum Circuit Simulator")
    print("="*70)
    
    simulator = QuantumCircuitSimulator()
    
    # Test parameters
    params = {
        'n_qubits': 2,
        'shots': 1000,
        'gate_sequence': 'bell'
    }
    
    # Profile execution
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple times
    n_runs = 50
    start_time = time.time()
    
    for _ in range(n_runs):
        result = simulator.run(params)
    
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    # Print statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print(f"\nTotal runs: {n_runs}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per run: {elapsed/n_runs*1000:.2f}ms")
    print(f"Throughput: {n_runs/elapsed:.2f} simulations/second")
    
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())
    
    return elapsed / n_runs


def profile_harmonic_oscillator():
    """Profile harmonic oscillator"""
    print("\n" + "="*70)
    print("PROFILING: Harmonic Oscillator")
    print("="*70)
    
    simulator = HarmonicOscillatorModule()
    
    # Test parameters
    params = {
        'oscillator_length': 1.0,
        'basis_size': 20,
        'initial_n': 5
    }
    
    # Profile execution
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple times
    n_runs = 100
    start_time = time.time()
    
    for _ in range(n_runs):
        result = simulator.run(params)
    
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    # Print statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print(f"\nTotal runs: {n_runs}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per run: {elapsed/n_runs*1000:.2f}ms")
    print(f"Throughput: {n_runs/elapsed:.2f} simulations/second")
    
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())
    
    return elapsed / n_runs


def identify_bottlenecks():
    """Identify optimization opportunities"""
    print("\n" + "="*70)
    print("BOTTLENECK ANALYSIS")
    print("="*70)
    
    # Profile each simulator
    schrodinger_time = profile_schrodinger()
    circuit_time = profile_quantum_circuit()
    harmonic_time = profile_harmonic_oscillator()
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    print(f"\nAverage execution times:")
    print(f"  Schrödinger Solver:     {schrodinger_time*1000:.2f}ms")
    print(f"  Quantum Circuit:        {circuit_time*1000:.2f}ms")
    print(f"  Harmonic Oscillator:    {harmonic_time*1000:.2f}ms")
    
    print(f"\nThroughput (simulations/second):")
    print(f"  Schrödinger Solver:     {1/schrodinger_time:.2f}")
    print(f"  Quantum Circuit:        {1/circuit_time:.2f}")
    print(f"  Harmonic Oscillator:    {1/harmonic_time:.2f}")
    
    # Identify slowest
    times = {
        'Schrödinger Solver': schrodinger_time,
        'Quantum Circuit': circuit_time,
        'Harmonic Oscillator': harmonic_time
    }
    
    slowest = max(times, key=times.get)
    fastest = min(times, key=times.get)
    
    print(f"\nSlowest simulator: {slowest} ({times[slowest]*1000:.2f}ms)")
    print(f"Fastest simulator: {fastest} ({times[fastest]*1000:.2f}ms)")
    print(f"Speed ratio: {times[slowest]/times[fastest]:.2f}x")
    
    # Optimization recommendations
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. Schrödinger Solver:")
    print("   - Already using CuPy for GPU acceleration")
    print("   - Consider reducing grid resolution for faster runs")
    print("   - Optimize FFT operations")
    print("   - Use in-place operations more aggressively")
    
    print("\n2. Quantum Circuit:")
    print("   - Consider GPU backend for Qiskit Aer")
    print("   - Reduce shots for faster execution (trade accuracy)")
    print("   - Cache circuit compilation")
    print("   - Batch multiple circuits together")
    
    print("\n3. Harmonic Oscillator:")
    print("   - Already very fast")
    print("   - Consider reducing basis_size if acceptable")
    print("   - Vectorize operations where possible")
    
    print("\n4. General:")
    print("   - Use multiprocessing for parallel simulations")
    print("   - Implement result caching for repeated parameters")
    print("   - Profile with larger datasets")
    print("   - Consider JIT compilation with numba")
    
    # Save profiling report
    report_path = Path('performance_reports')
    report_path.mkdir(exist_ok=True)
    
    with open(report_path / 'simulation_profiling.txt', 'w') as f:
        f.write("SIMULATION PERFORMANCE PROFILING\n")
        f.write("="*70 + "\n\n")
        f.write(f"Schrödinger Solver:     {schrodinger_time*1000:.2f}ms\n")
        f.write(f"Quantum Circuit:        {circuit_time*1000:.2f}ms\n")
        f.write(f"Harmonic Oscillator:    {harmonic_time*1000:.2f}ms\n")
        f.write(f"\nSlowest: {slowest}\n")
        f.write(f"Fastest: {fastest}\n")
        f.write(f"Speed ratio: {times[slowest]/times[fastest]:.2f}x\n")
    
    print(f"\n✓ Profiling report saved to: {report_path / 'simulation_profiling.txt'}")
    
    return True


if __name__ == '__main__':
    success = identify_bottlenecks()
    exit(0 if success else 1)
