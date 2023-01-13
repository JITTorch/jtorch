from audioop import mul
import jittor as jt
from jittor.dataset import Dataset as JDataset
from jittor.dataset import Sampler as JSampler, RandomSampler as JRandomSampler, SequentialSampler as JSequentialSampler
from typing import Any, Callable, Iterable, Optional, Sequence, Union
import jtorch.distributed as dist
import math

class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError


class DataLoader(JDataset):
    def __init__(self, dataset: Dataset, 
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False, 
                 sampler = None,
                 batch_sampler = None,
                 num_workers: int = 0, 
                 collate_fn = None,
                 pin_memory: bool = False, 
                 drop_last: bool = False,
                 timeout: float = 0, 
                 worker_init_fn = None,
                 multiprocessing_context=None, 
                 generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False,
                 pin_memory_device: str = "") -> None:
        super().__init__(batch_size=batch_size, 
                         shuffle=shuffle,
                         num_workers=num_workers,
                         drop_last=drop_last)
        
        unsupported_kwargs = {
            "sampler": sampler, 
            "batch_sampler": batch_sampler, 
            "pin_memory": pin_memory, 
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context, 
            "generator": generator, 
            "persistent_workers": persistent_workers, 
            "pin_memory_device": pin_memory_device
        }
        for kwarg, value in unsupported_kwargs.items():
            if value:
                jt.LOG.w(f"Not implemented Dataloader kwarg: {kwarg}")

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.total_len = 50000
        self.dataset.set_attrs(shuffle=shuffle)

    def collate_batch(self, batch):
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        else:
            return super().collate_batch(batch)

    def __getitem__(self, i):
        return self.dataset[i]    

# Sampler[T_co]
class DistributedSampler(JSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            indices = jt.randperm(len(self.dataset)).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

# for now
RandomSampler = JRandomSampler
SequentialSampler = JSequentialSampler