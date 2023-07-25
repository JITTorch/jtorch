import jittor as jt
import jittor.dataset
from jittor.dataset import Dataset as JDataset

from typing import Any, Callable, Iterable, Optional, Sequence, Union


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
        self.total_len = len(dataset)
        self.sampler = sampler

    def collate_batch(self, batch):
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        else:
            return super().collate_batch(batch)

    def __getitem__(self, i):
        return self.dataset[i]

class RandomSampler(jt.dataset.RandomSampler):
    def __init__(self, dataset, generator=None, **kwargs):
        super().__init__(dataset, **kwargs)

class DistributedSampler(jt.dataset.Sampler):
    def __init__(self, sampler: RandomSampler):
        assert(isinstance(sampler, RandomSampler))
        self.sampler = sampler

    def set_epoch(self, epoch: int):
        ### do nothing, let jittor's inner dataset handle 
        pass

    def __iter__(self):
        return self.sampler.__iter__()
    
    def __len__(self):
        return self.sampler.__len__()

BatchSampler = jt.dataset.BatchSampler
Sampler = jt.dataset.Sampler
SequentialSampler = jt.dataset.SequentialSampler
SubsetRandomSampler = jt.dataset.SubsetRandomSampler

IterableDataset = Dataset
TensorDataset = Dataset
