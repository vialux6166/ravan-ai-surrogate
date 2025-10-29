import h5py
import numpy as np

f = h5py.File('/home/windows/ravan-quantum-ml/data/raw/test_enhanced_qc.h5', 'r')

print('='*60)
print('ENHANCED QUANTUM CIRCUIT DATA ANALYSIS')
print('='*60)

print('\nFEATURES:')
for key in f['parameters'].keys():
    data = f['parameters'][key][:]
    print(f'\n{key}:')
    print(f'  Min: {data.min():.4f}, Max: {data.max():.4f}')
    print(f'  Mean: {data.mean():.4f}, Std: {data.std():.4f}')
    print(f'  Unique: {len(np.unique(data))}')

print('\n' + '='*60)
print('OBSERVABLES (Targets):')
print('='*60)
for key in f['observables'].keys():
    data = f['observables'][key][:]
    var = np.var(data)
    print(f'\n{key}:')
    print(f'  Min: {data.min():.6f}, Max: {data.max():.6f}')
    print(f'  Mean: {data.mean():.6f}, Std: {data.std():.6f}')
    print(f'  Variance: {var:.6f}')
    if var < 0.001:
        print(f'  ⚠️  CONSTANT (var < 0.001)')
    else:
        print(f'  ✓ LEARNABLE')

print('\n' + '='*60)
print('SUMMARY:')
print('='*60)
learnable = sum(1 for key in f['observables'].keys() if np.var(f['observables'][key][:]) >= 0.001)
total = len(f['observables'].keys())
print(f'Learnable targets: {learnable}/{total}')
print(f'Samples: {len(f["observables"][list(f["observables"].keys())[0]])}')

f.close()
