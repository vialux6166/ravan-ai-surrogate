"""
Deploy Python files to WSL
"""
import subprocess
import sys
from pathlib import Path

files_to_deploy = [
    ('ravan_logging_config.py', 'src/utils/logging.py'),
    ('gpu_accelerator.py', 'src/gpu/accelerator.py'),
    ('vram_manager.py', 'src/gpu/vram_manager.py'),
]

wsl_base = '/home/windows/ravan-quantum-ml'

for src, dst in files_to_deploy:
    src_path = Path(src)
    if not src_path.exists():
        print(f"Error: {src} not found")
        continue
    
    # Read file content
    with open(src_path, 'r') as f:
        content = f.read()
    
    # Write to WSL using wsl command
    wsl_dst = f"{wsl_base}/{dst}"
    cmd = f'wsl bash -c "cat > {wsl_dst}" '
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=content)
        
        if process.returncode == 0:
            print(f"✓ Deployed: {src} → {dst}")
        else:
            print(f"✗ Failed: {src} - {stderr}")
    except Exception as e:
        print(f"✗ Error deploying {src}: {e}")

print("\nDeployment complete!")
