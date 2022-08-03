from typing import Any, Optional
from pathlib import Path
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .mlm import MBM


class OfflineTrajectoryDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        n_agents: Optional[int] = 3,
        episode_length: Optional[int] = 151,
        **kwargs
    ):
        files = Path(root_dir).glob("*.h5")
        self.path = [f for f in files][0]
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.mbm = MBM(**kwargs)

    def __len__(self):
        num_episodes = 0
        with h5py.File(self.path, "r") as f:
            num_episodes = len(f["actions"])
        return num_episodes

    def __getitem__(self, index: Any):
        """
        inputs:
            t1 = [(o, a), ...]
            t2 = [(o, a), ...]
            batch = stack(t1, t2, ...)
        """
        with h5py.File(self.path, "r") as f:
            a_onehot = torch.tensor(f["actions_onehot"])[index]
            o = torch.tensor(f["obs"])[index]
            d = torch.tensor(f["terminated"])[index]

        t = torch.zeros((self.episode_length, 1), dtype=torch.int32)
        terminate_idx = torch.argmax(d)
        t[:terminate_idx, 0] = torch.arange(terminate_idx)

        o = o.reshape(self.episode_length * self.n_agents, -1)
        a_onehot = a_onehot.reshape(self.episode_length * self.n_agents, -1)
        # convert onehot actions to action index tokens
        a_tokens = torch.argmax(a_onehot, dim=1)

        # get masked behavior modeling mask
        a_tokens, y = self.mbm(a_tokens)

        a_onehot = F.one_hot(a_tokens)

        x = torch.cat([o, a_onehot], dim=1)

        return x, t, y


if __name__ == "__main__":
    dataset = OfflineTrajectoryDataset("offline_data/")
    sample = dataset.__getitem__(0)
