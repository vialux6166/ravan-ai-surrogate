"""
Compatibility shims for legacy import paths used in tests.

Avoid circular imports by importing HDF5Storage lazily.
"""

from dataset import Dataset  # noqa: F401

def __getattr__(name):
    if name == 'HDF5Storage':
        # Lazy import to avoid circular reference when hdf5_storage imports pipeline
        from hdf5_storage import HDF5Storage
        return HDF5Storage
    raise AttributeError(name)


