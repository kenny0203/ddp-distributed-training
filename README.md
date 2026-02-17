# 🔥 PyTorch Distributed Training (DDP) – CPU Multi-Process Simulation

This project demonstrates Distributed Data Parallel (DDP) training using PyTorch.

It simulates multi-GPU training using multiple CPU processes via `torchrun`.

---

## 🚀 Features

- Multi-process distributed training
- Dataset sharding with `DistributedSampler`
- Gradient synchronization using `all_reduce`
- Rank-based logging (only rank 0 prints)
- Simulated multi-GPU setup using CPU

---

## 🧠 How It Works

1. `torchrun` spawns N processes
2. Each process:
   - Initializes a distributed process group
   - Loads a shard of the dataset
   - Wraps model with `DistributedDataParallel`
3. Gradients are synchronized across processes
4. Loss is averaged across ranks

---

## ▶️ Run (2 processes)

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 ddp_train.py

▶️ Run (4 processes)

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 ddp_train.py


📦 Requirements
pip install torch torchvision


```markdown
## 🏷 Technologies

- Python
- PyTorch
- DistributedDataParallel (DDP)
- torchrun
- WSL2
