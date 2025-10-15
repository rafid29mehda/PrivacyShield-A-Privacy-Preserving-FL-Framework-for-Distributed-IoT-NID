# PrivacyShield-A-Privacy-Preserving-FL-Framework-for-Distributed-IoT-NID

## Overview

PrivacyShield is a federated learning system that enables collaborative training of intrusion detection models across distributed IoT networks while preserving data privacy through differential privacy mechanisms. The system leverages the TON-IoT dataset to detect multiple types of network attacks without centralizing sensitive network traffic data.

## Key Features

- **Decentralized Training**: 10 virtual clients train collaboratively without sharing raw data
- **Differential Privacy**: Opacus integration ensures gradient-level privacy protection
- **Non-IID Data Distribution**: Realistic heterogeneous data partitioning across clients
- **Multi-Class Classification**: Detects various network intrusion types
- **Scalable Architecture**: Built on Flower framework for efficient FL orchestration
- **GPU Acceleration**: CUDA support for faster model training

## Architecture

The system implements a three-layer neural network architecture optimized for tabular network traffic data:
- Input Layer: Dynamic feature size based on preprocessed network attributes
- Hidden Layers: 128 → 64 neurons with ReLU activation
- Output Layer: Multi-class softmax for attack type classification

## Requirements

```
Python >= 3.8
torch >= 1.9.0
flwr[simulation] >= 1.0.0
opacus >= 1.0.0
scikit-learn >= 0.24.0
pandas >= 1.3.0
numpy >= 1.20.0
```


## Dataset

This project uses the **TON-IoT Network Dataset** from the University of New South Wales (UNSW).

**Dataset Details:**
- Source: TON-IoT Telemetry Dataset
- Type: Network traffic flows
- Size: ~450MB
- Features: Network flow attributes (IPs, ports, protocols, duration, bytes)
- Labels: Multiple attack types (Normal, DDoS, DoS, Injection, etc.)


### Data Preprocessing Pipeline

1. **Missing Value Handling**: Median imputation for numeric features
2. **Categorical Encoding**: Label encoding for non-numeric attributes
3. **Feature Scaling**: StandardScaler normalization (μ=0, σ=1)
4. **Data Partitioning**: Non-IID distribution with label skew
5. **Type Conversion**: NumPy float32 arrays for PyTorch compatibility

### Federated Learning Workflow

1. **Initialization**: Server distributes initial global model to clients
2. **Client Selection**: Randomly sample 50% of clients per round
3. **Local Training**: Each client trains for 2 epochs with DP-SGD
4. **Aggregation**: FedAvg algorithm combines client updates
5. **Evaluation**: Test global model on client validation sets
6. **Iteration**: Repeat for 20 rounds

### Privacy Mechanism

Differential privacy is enforced through:
- **Per-sample gradient clipping**: Maximum L2 norm = 1.0
- **Gaussian noise addition**: Noise multiplier σ = 1.0
- **Privacy accounting**: Automatic ε-δ budget tracking via Opacus

