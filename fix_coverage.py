"""Fix parameter coverage calculation for boolean parameters"""

import sys

# Read from WSL via stdin simulation
fix_code = """
import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

with open('src/pipeline/parameter_sweep.py', 'r') as f:
    content = f.read()

# Fix the coverage calculation to handle constant ranges (booleans)
old_calc = '''            coverage = (actual_max - actual_min) / (max_val - min_val)'''

new_calc = '''            range_val = max_val - min_val
            if range_val == 0:
                coverage = 1.0  # Boolean or constant parameter
            else:
                coverage = (actual_max - actual_min) / range_val'''

content = content.replace(old_calc, new_calc)

with open('src/pipeline/parameter_sweep.py', 'w') as f:
    f.write(content)

print('Fixed parameter_sweep.py coverage calculation')
"""

print(fix_code)
