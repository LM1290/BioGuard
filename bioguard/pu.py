"""Positive-unlabeled learning primitives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class NonNegativePULoss(nn.Module):
    """Non-negative positive-unlabeled risk for a supplied class prior."""

    def __init__(self, class_prior: float):
        super().__init__()
        if not 0.0 < class_prior < 1.0:
            raise ValueError("class_prior must be within (0, 1)")
        self.class_prior = float(class_prior)

    def forward(self, logits, targets):
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        positive_mask = targets == 1
        unlabeled_mask = targets == 0
        if not positive_mask.any() or not unlabeled_mask.any():
            raise ValueError(
                "Every nnPU optimization batch requires positive and unlabeled samples"
            )

        positive_logits = logits[positive_mask]
        unlabeled_logits = logits[unlabeled_mask]
        positive_risk = self.class_prior * F.softplus(-positive_logits).mean()
        negative_risk = (
            F.softplus(unlabeled_logits).mean()
            - self.class_prior * F.softplus(positive_logits).mean()
        )
        return positive_risk + torch.clamp(negative_risk, min=0.0)


class PUBalancedBatchSampler(Sampler):
    """Yield deterministic batches containing both PU observation classes."""

    def __init__(self, targets, batch_size: int, seed: int):
        if batch_size < 2:
            raise ValueError("PU batch_size must be at least 2")
        target_array = np.asarray(targets)
        self.positive = np.flatnonzero(target_array == 1)
        self.unlabeled = np.flatnonzero(target_array == 0)
        if not len(self.positive) or not len(self.unlabeled):
            raise ValueError("PU training requires positive and unlabeled samples")
        self.batch_size = batch_size
        self.positive_per_batch = max(1, batch_size // 2)
        self.unlabeled_per_batch = batch_size - self.positive_per_batch
        self.seed = seed
        self.epoch = 0
        self.n_batches = max(
            int(np.ceil(len(self.positive) / self.positive_per_batch)),
            int(np.ceil(len(self.unlabeled) / self.unlabeled_per_batch)),
        )

    @staticmethod
    def _draw(indices, count, rng):
        chunks = []
        remaining = count
        while remaining:
            shuffled = rng.permutation(indices)
            take = min(remaining, len(shuffled))
            chunks.append(shuffled[:take])
            remaining -= take
        return np.concatenate(chunks)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        positive = self._draw(
            self.positive,
            self.n_batches * self.positive_per_batch,
            rng,
        )
        unlabeled = self._draw(
            self.unlabeled,
            self.n_batches * self.unlabeled_per_batch,
            rng,
        )
        for batch_index in range(self.n_batches):
            pos_start = batch_index * self.positive_per_batch
            unl_start = batch_index * self.unlabeled_per_batch
            batch = np.concatenate(
                [
                    positive[
                        pos_start:pos_start + self.positive_per_batch
                    ],
                    unlabeled[
                        unl_start:unl_start + self.unlabeled_per_batch
                    ],
                ]
            )
            yield rng.permutation(batch).tolist()

    def __len__(self):
        return self.n_batches
