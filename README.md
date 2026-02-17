# Distributed Training with PyTorch DDP (CPU Simulation)

This project demonstrates DistributedDataParallel (DDP) training using PyTorch.

## Features
- Multi-process distributed training
- Dataset sharding with DistributedSampler
- Gradient synchronization using all_reduce
- Simulated multi-GPU via CPU processes

## Run

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 ddp_train.py
```

## Requirements

```bash
pip install -r requirements.txt
```
