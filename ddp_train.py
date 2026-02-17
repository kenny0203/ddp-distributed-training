import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ToyDataset(Dataset):
    def __init__(self, n=8000, d=128):
        self.x = torch.randn(n, d)
        self.y = (self.x.sum(dim=1) > 0).long()
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]

def main():
    dist.init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 2),
    ).to(device)
    ddp = DDP(model)

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=0)

    opt = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        sampler.set_epoch(epoch)
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(ddp(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = torch.tensor(total / len(loader))
        dist.all_reduce(avg, op=dist.ReduceOp.SUM)
        avg = (avg / world_size).item()
        if rank == 0:
            print(f"epoch {epoch} | avg_loss {avg:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
