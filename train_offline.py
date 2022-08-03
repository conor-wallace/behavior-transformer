import argparse
from types import SimpleNamespace as SN

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.behavior_transformer.datamodule import OfflineTrajectoryDataModule
from src.behavior_transformer.module import BehaviorCloningClassifierModule


def get_params():
    config = {
        "n_agents": 3,
        "n_actions": 9,
        "episode_limit": 150,
        "token_embed_dim": 32,
        "hidden_ff": 32,
        "dropout": 0.2,
        "batch_size": 4,
    }
    return SN(**config)


def train(args):
    params = get_params()
    datamodule = OfflineTrajectoryDataModule(
        args.root_dir, n_actions=params.n_actions, batch_size=params.batch_size
    )
    module = BehaviorCloningClassifierModule(args.shape, params)

    if args.log == "wandb":
        logger = WandbLogger(project="Behavior Transformer")
    else:
        logger = None

    trainer = pl.Trainer(logger=logger, log_every_n_steps=1, enable_model_summary=True)
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        nargs="?",
        type=str,
        default="src/behavior_transformer/offline_data/",
    )
    parser.add_argument("--shape", nargs="?", type=int, default=45)
    parser.add_argument("--log", nargs="?", type=str)

    args = parser.parse_args()
    train(args)
