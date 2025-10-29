"""
Shim module for legacy import path `simulators.harmonic`.
Re-exports HarmonicOscillator from root-level `harmonic_oscillator.py`.
"""

from harmonic_oscillator import HarmonicOscillator  # noqa: F401

# Legacy alias expected by tests
HarmonicOscillatorModule = HarmonicOscillator


