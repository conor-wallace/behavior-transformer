from typing import List, Optional

import torch
import torch.nn as nn


class MBM(nn.Module):
    """
    MBM: Masked Behavior Modeling

    This module takes in a sequence of timesteps and returns the attention mask
    and corresponding ground truth mask for self-supervised training of a BERT-based behavior model.
    """

    def __init__(
        self,
        n_actions: int,  # replace this with a range of values that represent valid action tokens for continuous environments
        no_mask_tokens: Optional[List[int]] = [],
        masking_prob: Optional[float] = 0.15,
        random_prob: Optional[float] = 0.1,
        original_prob: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        self.n_tokens = n_actions
        self.mask_token = n_actions + 1
        self.pad_token = 0
        self.no_mask_tokens = no_mask_tokens + [self.pad_token, self.mask_token]
        self.mask_prob = masking_prob
        self.rand_prob = random_prob
        self.orig_prob = original_prob

    def forward(self, x: torch.Tensor):
        full_mask = torch.rand(x.shape) < self.mask_prob

        for t in self.no_mask_tokens:
            full_mask &= x != t

        orig_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.orig_prob)
        rand_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.rand_prob)
        rand_idxs = torch.nonzero(rand_mask, as_tuple=True)
        random_tokens = torch.randint(
            0, self.n_tokens, (len(rand_idxs[0]),), device=x.device
        )
        mask = full_mask & ~rand_mask & ~orig_mask
        y = x.clone()

        x.masked_fill_(mask, self.mask_token)
        x[rand_idxs] = random_tokens

        y.masked_fill_(~full_mask, self.pad_token)

        return x, y


if __name__ == "__main__":
    n_actions = 9
    mbm = MBM(
        n_actions=n_actions,
    )

    x = torch.randint(low=1, high=n_actions + 1, size=(25, 1))
    x, y = mbm(x)
