from contextlib import contextmanager
import torch
import torch.distributed as dist


# NOTE. for multi-gpu train in windows use gloo backend
@contextmanager
def custom_torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        # NOTE. modified by ryanwfu 2023/09/25
        if dist.get_backend() == 'nccl':
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
    yield
    if initialized and local_rank == 0:
        # NOTE. modified by ryanwfu 2023/09/25
        if dist.get_backend() == 'nccl':
            dist.barrier(device_ids=[0])
        else:
            dist.barrier()