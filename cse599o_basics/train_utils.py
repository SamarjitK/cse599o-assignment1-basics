import torch
import numpy.typing as npt
import os
from typing import BinaryIO, IO

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    states: dict = torch.load(src)
    model.load_state_dict(states['model_state_dict'])
    optimizer.load_state_dict(states['optimizer_state_dict'])
    return states['iteration']

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    """Dump all state from model, optimizer, and iteration into `out`."""
    # Use state_dict() for both model and optimizer
    # Use torch.save(obj, out) to dump obj into out
    states = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(states, out)

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]
    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    labels = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    for i in range(batch_size):
        start = torch.randint(0, n - context_length, (1,)).item()
        inputs[i] = torch.from_numpy(dataset[start : start + context_length])
        labels[i] = torch.from_numpy(dataset[start + 1 : start + context_length + 1])
    return inputs, labels