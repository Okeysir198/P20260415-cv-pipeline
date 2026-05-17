"""Multi-task dataset wrapper for detection training.

Yields single-task batches via consecutive-yield sampling: pick a task, emit
``batch_size`` samples from it, repeat. This guarantees a collated batch is
always single-task — the model's cls-head routing requires it.

Per-task validation datasets are built separately as plain ``YOLOXDataset``s
(no interleaving), so per-task mAP can be reported in the eval loop.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset


class MultitaskInterleaver(IterableDataset):
    """Interleaves N detection datasets so each batch lands on one task.

    Sampling strategies (per-task probabilities):
      - ``round_robin_sqrt``: p_i ∝ sqrt(len(ds_i))   (default; protects small tasks)
      - ``uniform``:          p_i = 1/N
      - ``proportional``:     p_i ∝ len(ds_i)

    Iteration yields tuples of (image, target, task_name, path), where target
    is whatever the underlying ``YOLOXDataset.__getitem__`` returns (its
    transformed numpy/tensor target). Use :func:`multitask_collate_fn` to
    batch these into the HF DETR format.

    The interleaver does NOT shuffle within a task on its own — it relies on
    each underlying dataset's standard shuffling (DataLoader's
    ``shuffle=True`` doesn't apply to IterableDataset; we shuffle the index
    list once per epoch instead).

    Length is reported as the sum of per-task dataset lengths so HF Trainer's
    progress bar and epoch accounting work correctly.
    """

    def __init__(
        self,
        task_datasets: dict,
        batch_size: int,
        strategy: str = "round_robin_sqrt",
        weights: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not task_datasets:
            raise ValueError("task_datasets cannot be empty")
        self.task_datasets = task_datasets
        self.batch_size = int(batch_size)
        self.strategy = strategy
        self.seed = int(seed)
        self.task_names = list(task_datasets.keys())

        lens = {n: len(ds) for n, ds in task_datasets.items()}
        if weights:
            w = {n: float(weights.get(n, 1.0)) for n in self.task_names}
        elif strategy == "uniform":
            w = {n: 1.0 for n in self.task_names}
        elif strategy == "proportional":
            w = {n: float(lens[n]) for n in self.task_names}
        else:  # round_robin_sqrt (default)
            w = {n: math.sqrt(max(1, lens[n])) for n in self.task_names}
        total = sum(w.values()) or 1.0
        self.task_probs = {n: w[n] / total for n in self.task_names}

        self._total_len = sum(lens.values())
        self._epoch = 0

    def __len__(self) -> int:
        return self._total_len

    def _shuffled_indices(self, rng: random.Random) -> dict[str, list[int]]:
        out = {}
        for name, ds in self.task_datasets.items():
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            out[name] = idx
        return out

    def __iter__(self) -> Iterator[tuple]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = self.seed + self._epoch * 997 + worker_id
        self._epoch += 1
        rng = random.Random(seed)

        cursors = {n: 0 for n in self.task_names}
        per_task_idx = self._shuffled_indices(rng)
        names = self.task_names
        probs = [self.task_probs[n] for n in names]

        target_total = self._total_len
        emitted = 0
        while emitted < target_total:
            # Pick a task for the next mini-batch.
            task = rng.choices(names, weights=probs, k=1)[0]
            ds = self.task_datasets[task]
            ds_len = len(ds)
            for _ in range(self.batch_size):
                if emitted >= target_total:
                    return
                if cursors[task] >= ds_len:
                    # Re-shuffle that task's index list and wrap around.
                    rng.shuffle(per_task_idx[task])
                    cursors[task] = 0
                idx = per_task_idx[task][cursors[task]]
                cursors[task] += 1
                sample = ds[idx]
                # YOLOXDataset returns (image, target, path)
                if len(sample) == 3:
                    image, target, path = sample
                else:
                    image, target = sample[0], sample[1]
                    path = ""
                yield image, target, task, path
                emitted += 1


class TaskLabeledDataset(torch.utils.data.Dataset):
    """Wraps a raw detection dataset so each sample carries its task_name.

    YOLOXDataset.__getitem__ returns (image, target, path); the multitask
    collator requires (image, target, task_name, path). This wrapper injects
    task_name without copying data.
    """

    def __init__(self, dataset, task_name: str):
        self._ds = dataset
        self._task = task_name

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]
        if len(sample) == 3:
            image, target, path = sample
        elif len(sample) == 2:
            image, target = sample
            path = ""
        else:
            raise ValueError(f"Unexpected sample arity: {len(sample)}")
        return image, target, self._task, path


def multitask_collate_fn(batch: list) -> dict:
    """Collate single-task samples into the HF DETR batch dict.

    Enforces single-task invariant — if a batch mixes tasks (which the
    interleaver should never produce), raises a clear error.

    Output:
        {
          "pixel_values": Tensor(B, 3, H, W),
          "labels":       list[{class_labels, boxes}],
          "task_name":    str
        }
    """
    if not batch:
        raise ValueError("Empty batch")
    task_names = {item[2] for item in batch}
    if len(task_names) != 1:
        raise ValueError(
            f"Mixed-task batch is unsupported (got tasks={task_names}); "
            "MultitaskInterleaver should yield single-task batches."
        )
    task = next(iter(task_names))

    images = torch.stack([item[0] for item in batch])
    labels = []
    for _, targets, _t, _p in batch:
        if hasattr(targets, "class_labels"):
            labels.append(targets)
        elif isinstance(targets, torch.Tensor) and targets.numel() == 0:
            labels.append({
                "class_labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros(0, 4, dtype=torch.float32),
            })
        else:
            labels.append({
                "class_labels": targets[:, 0].long(),
                "boxes": targets[:, 1:5].float(),
            })
    return {"pixel_values": images, "labels": labels, "task_name": task}
