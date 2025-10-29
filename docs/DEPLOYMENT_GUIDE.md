# Deployment Guide

## Overview

This guide covers deploying the Ravan Quantum-ML System in various environments.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring](#monitoring)
6. [Security](#security)
7. [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Prerequisites

- Ubuntu 22.04 LTS (or WSL2)
- NVIDIA GPU with 8+ GB VRAM
- CUDA 12.x
- Python 3.10+

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ravan.git
cd ravan
```

### Step 2: Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python verify_installation.py
```

### Step 4: Start API Server

```bash
python api_server.py
```

Server will be available at `http://localhost:8000`

### Step 5: Test API

```bash
curl http://localhost:8000/health
```

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- NVIDIA Docker runtime
- NVIDIA GPU with drivers

### Step 1: Install NVIDIA Docker

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Step 2: Build Docker Image

```bash
docker build -t ravan:latest .
```

### Step 3: Run Container

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name ravan-api \
  ravan:latest
```

### Step 4: Test Container

```bash
curl http://localhost:8000/health
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ravan-api

# Stop services
docker-compose down
```

---

## Cloud Deployment

### AWS Deployment

#### Option 1: EC2 with GPU

**1. Launch EC2 Instance**

- Instance type: `g4dn.xlarge` or `p3.2xlarge`
- AMI: Deep Learning AMI (Ubuntu 22.04)
- Storage: 100 GB EBS
- Security group: Allow ports 22, 8000

**2. Connect and Setup**

```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone repository
git clone https://github.com/yourusername/ravan.git
cd ravan

# Setup environment
./setup.sh

# Start server
python api_server.py
```

**3. Configure Load Balancer**

- Create Application Load Balancer
- Target group: Port 8000
- Health check: `/health`
- SSL certificate: AWS Certificate Manager

#### Option 2: ECS with Fargate

**1. Create ECR Repository**

```bash
aws ecr create-repository --repository-name ravan
```

**2. Push Docker Image**

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag ravan:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ravan:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ravan:latest
```

**3. Create ECS Task Definition**

```json
{
  "family": "ravan-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "ravan-api",
      "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ravan:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ]
    }
  ]
}
```

### GCP Deployment

#### Using Compute Engine with GPU

**1. Create Instance**

```bash
gcloud compute instances create ravan-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

**2. Install NVIDIA Drivers**

```bash
gcloud compute ssh ravan-instance

# Install drivers
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers
```

**3. Deploy Application**

```bash
# Clone and setup
git clone https://github.com/yourusername/ravan.git
cd ravan
./setup.sh

# Start server
python api_server.py
```

### Azure Deployment

#### Using Azure Container Instances with GPU

**1. Create Resource Group**

```bash
az group create --name ravan-rg --location eastus
```

**2. Create Container Registry**

```bash
az acr create --resource-group ravan-rg \
  --name ravanregistry --sku Basic
```

**3. Push Image**

```bash
az acr login --name ravanregistry
docker tag ravan:latest ravanregistry.azurecr.io/ravan:latest
docker push ravanregistry.azurecr.io/ravan:latest
```

**4. Deploy Container**

```bash
az container create \
  --resource-group ravan-rg \
  --name ravan-api \
  --image ravanregistry.azurecr.io/ravan:latest \
  --cpu 4 \
  --memory 16 \
  --gpu-count 1 \
  --gpu-sku V100 \
  --ports 8000 \
  --dns-name-label ravan-api
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with GPU support
- kubectl configured
- NVIDIA device plugin installed

### Step 1: Install NVIDIA Device Plugin

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

### Step 2: Create Deployment

**deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ravan-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ravan-api
  template:
    metadata:
      labels:
        app: ravan-api
    spec:
      containers:
      - name: ravan-api
        image: ravan:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: models
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ravan-models-pvc
```

### Step 3: Create Service

**service.yaml**:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ravan-api-service
spec:
  type: LoadBalancer
  selector:
    app: ravan-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### Step 4: Deploy

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Step 5: Verify

```bash
kubectl get pods
kubectl get services
kubectl logs -f deployment/ravan-api
```

---

## Monitoring

### Prometheus + Grafana

**1. Deploy Monitoring Stack**

```bash
docker-compose up -d prometheus grafana
```

**2. Access Dashboards**

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

**3. Configure Grafana**

- Add Prometheus data source
- Import dashboard from `monitoring/grafana-dashboards/`

### Custom Metrics

The API exposes metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Available Metrics**:
- Request count
- Request latency
- GPU memory usage
- Model inference time
- Simulation time

### Logging

**View Logs**:

```bash
# Docker
docker logs -f ravan-api

# Kubernetes
kubectl logs -f deployment/ravan-api

# Local
tail -f logs/ravan.log
```

**Log Levels**:
- `DEBUG`: Detailed information
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

---

## Security

### API Authentication

**1. Generate API Token**

```bash
python generate_token.py
```

**2. Use Token in Requests**

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/predict
```

### HTTPS/TLS

**1. Generate SSL Certificate**

```bash
# Self-signed (development)
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes

# Production: Use Let's Encrypt
certbot certonly --standalone -d your-domain.com
```

**2. Configure Server**

```python
# In api_server.py
uvicorn.run(
    "api_server:app",
    host="0.0.0.0",
    port=443,
    ssl_keyfile="key.pem",
    ssl_certfile="cert.pem"
)
```

### Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### Rate Limiting

Configure in `api_server.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, ...):
    ...
```

---

## Troubleshooting

### GPU Not Detected

**Check NVIDIA drivers**:
```bash
nvidia-smi
```

**Check Docker GPU access**:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

**Reduce batch size**:
```python
config.hyperparameters['batch_size'] = 32
```

**Enable optimizations**:
```python
use_checkpointing = True
use_mixed_precision = True
```

### Slow Inference

**Check GPU utilization**:
```bash
nvidia-smi -l 1
```

**Enable TorchScript**:
```python
model = torch.jit.script(model)
```

### Container Won't Start

**Check logs**:
```bash
docker logs ravan-api
```

**Verify GPU access**:
```bash
docker run --rm --gpus all ravan:latest nvidia-smi
```

---

## Performance Tuning

### Optimize for Throughput

```python
# Increase workers
uvicorn.run(..., workers=4)

# Increase batch size
config.hyperparameters['batch_size'] = 256

# Enable mixed precision
use_mixed_precision = True
```

### Optimize for Latency

```python
# Use smaller model
config.hyperparameters['hidden_dims'] = [128, 64]

# Reduce batch size
config.hyperparameters['batch_size'] = 1

# Use TorchScript
model = torch.jit.script(model)
```

---

## Backup and Recovery

### Backup Models

```bash
# Backup models directory
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Upload to S3
aws s3 cp models-backup-*.tar.gz s3://your-bucket/backups/
```

### Restore Models

```bash
# Download from S3
aws s3 cp s3://your-bucket/backups/models-backup-20251019.tar.gz .

# Extract
tar -xzf models-backup-20251019.tar.gz
```

---

## Scaling

### Horizontal Scaling

**Kubernetes**:
```bash
kubectl scale deployment ravan-api --replicas=5
```

**Docker Swarm**:
```bash
docker service scale ravan-api=5
```

### Vertical Scaling

**Increase resources**:
```yaml
resources:
  limits:
    nvidia.com/gpu: 2  # Use 2 GPUs
    memory: "32Gi"
    cpu: "8"
```

---

## Support

For deployment issues:
- Check logs first
- Review troubleshooting section
- Open GitHub issue
- Contact support@ravan-ml.com

---

**Last Updated**: 2025-10-19  
**Version**: 1.0.0
