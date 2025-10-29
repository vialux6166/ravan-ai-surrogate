# Docker Deployment Guide

This directory contains Docker configuration for the Ravan Quantum-ML System.

## Quick Start

### Build Image
```bash
# Build without LLM support (smaller image)
docker build -t ravan-quantum-ml:latest .

# Build with LLM support (requires 20GB+ VRAM)
# Uncomment LLM installation line in Dockerfile first
docker build -t ravan-quantum-ml:llm .
```

### Run Container
```bash
# Run with GPU support
docker run --gpus all -it ravan-quantum-ml:latest bash

# Run tests
docker run --gpus all ravan-quantum-ml:latest python3 -m pytest tests/ -v

# Run simulation
docker run --gpus all ravan-quantum-ml:latest python3 -c "from ravan import simulate; print(simulate({'simulator': 'schrodinger', 'V0': 5.0}))"
```

### Using Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build
```

## Configuration

### GPU Support

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- Docker version 19.03+

**Install NVIDIA Docker:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Verify GPU Access:**
```bash
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Environment Variables

Set in `docker-compose.yml` or pass with `-e`:

- `CUDA_VISIBLE_DEVICES` - GPU device ID (default: 0)
- `PYTHONUNBUFFERED` - Python output buffering (default: 1)
- `RAVAN_ENV` - Environment (development/production)

### Volumes

Persistent data directories:
- `./data` → `/app/data` - Datasets
- `./models` → `/app/models` - Trained models
- `./results` → `/app/results` - Results
- `./logs` → `/app/logs` - Log files

### Resource Limits

Configured in `docker-compose.yml`:
- Memory: 32GB
- Shared memory: 8GB (for PyTorch)
- GPU: 1 device

## Usage Examples

### Interactive Development
```bash
# Start interactive shell
docker run --gpus all -it -v $(pwd):/app ravan-quantum-ml:latest bash

# Inside container
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
pytest tests/ -v
```

### Run Simulations
```bash
# Quantum circuit
docker run --gpus all ravan-quantum-ml:latest \
    python3 -c "from ravan import simulate; print(simulate({'simulator': 'quantum_circuit', 'n_qubits': 2}))"

# Schrödinger solver
docker run --gpus all ravan-quantum-ml:latest \
    python3 -c "from ravan import simulate; print(simulate({'simulator': 'schrodinger', 'V0': 5.0}))"
```

### Train Models
```bash
# Mount data directory and train
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    ravan-quantum-ml:latest \
    python3 -c "from ravan import train; train('data/dataset.h5', 'mlp', 'models/mlp_model')"
```

### Run Tests
```bash
# All tests
docker run --gpus all ravan-quantum-ml:latest pytest tests/ -v

# Specific test
docker run --gpus all ravan-quantum-ml:latest pytest tests/test_code_sandbox.py -v

# With coverage
docker run --gpus all ravan-quantum-ml:latest pytest tests/ --cov=. --cov-report=html
```

## Image Variants

### Base Image (No LLM)
- **Size**: ~8GB
- **Features**: Core simulations, ML models
- **VRAM**: 8GB+ recommended
- **Build**: Default Dockerfile

### LLM Image
- **Size**: ~25GB
- **Features**: Core + LLM integration
- **VRAM**: 20GB+ required
- **Build**: Uncomment LLM line in Dockerfile

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA Docker
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker version
docker --version  # Need 19.03+

# Restart Docker
sudo systemctl restart docker
```

### Out of Memory
```bash
# Increase shared memory
docker run --gpus all --shm-size=16g ravan-quantum-ml:latest

# Or in docker-compose.yml
shm_size: 16g
```

### Permission Issues
```bash
# Run as current user
docker run --gpus all --user $(id -u):$(id -g) ravan-quantum-ml:latest

# Fix volume permissions
sudo chown -R $(id -u):$(id -g) data/ models/ results/
```

### Build Failures
```bash
# Clean build
docker build --no-cache -t ravan-quantum-ml:latest .

# Check disk space
df -h

# Prune old images
docker system prune -a
```

## Production Deployment

### Best Practices
1. Use specific version tags (not `latest`)
2. Set resource limits
3. Configure health checks
4. Use secrets for sensitive data
5. Enable logging
6. Set restart policies

### Example Production Config
```yaml
services:
  ravan:
    image: ravan-quantum-ml:1.0.0
    restart: always
    mem_limit: 32g
    healthcheck:
      test: ["CMD", "python3", "-c", "import torch"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Security
```bash
# Run as non-root user
RUN useradd -m -u 1000 ravan
USER ravan

# Scan for vulnerabilities
docker scan ravan-quantum-ml:latest
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Docker Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t ravan-quantum-ml:${{ github.sha }} .
      - name: Run tests
        run: docker run ravan-quantum-ml:${{ github.sha }} pytest tests/
```

## Maintenance

### Update Dependencies
```bash
# Rebuild with latest packages
docker build --pull --no-cache -t ravan-quantum-ml:latest .
```

### Clean Up
```bash
# Remove old containers
docker container prune

# Remove old images
docker image prune -a

# Remove all unused data
docker system prune -a --volumes
```

## Support

For issues or questions:
- Check logs: `docker logs ravan-quantum-ml`
- Inspect container: `docker inspect ravan-quantum-ml`
- Access shell: `docker exec -it ravan-quantum-ml bash`
