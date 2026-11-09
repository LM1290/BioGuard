import unittest

import numpy as np
import torch

from bioguard.pu import NonNegativePULoss, PUBalancedBatchSampler


class PositiveUnlabeledTests(unittest.TestCase):
    def test_nnpu_loss_matches_risk_definition(self):
        logits = torch.tensor([1.2, -0.4, 0.3, -1.1])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        prior = 0.2

        positive = logits[:2]
        unlabeled = logits[2:]
        expected_positive = prior * torch.nn.functional.softplus(
            -positive
        ).mean()
        expected_negative = (
            torch.nn.functional.softplus(unlabeled).mean()
            - prior * torch.nn.functional.softplus(positive).mean()
        )
        expected = expected_positive + torch.clamp(
            expected_negative,
            min=0.0,
        )

        actual = NonNegativePULoss(prior)(logits, targets)
        torch.testing.assert_close(actual, expected)

    def test_nnpu_rejects_single_observation_class(self):
        loss = NonNegativePULoss(0.1)
        with self.assertRaisesRegex(ValueError, "positive and unlabeled"):
            loss(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))

    def test_balanced_sampler_puts_both_classes_in_every_batch(self):
        targets = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        sampler = PUBalancedBatchSampler(targets, batch_size=4, seed=5)

        for batch in sampler:
            batch_targets = set(targets[batch])
            self.assertEqual(batch_targets, {0, 1})
            self.assertEqual(len(batch), 4)

    def test_balanced_sampler_is_reproducible(self):
        targets = np.array([1, 1, 0, 0, 0])
        first = list(PUBalancedBatchSampler(targets, 3, seed=9))
        second = list(PUBalancedBatchSampler(targets, 3, seed=9))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
