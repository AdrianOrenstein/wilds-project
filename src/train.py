from argparse import ArgumentParser
import time
from typing import Any, Dict
import warnings

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.experiments.experiments import get_experiment
import torch

PROJECT_NAME = "adrian-wilds-project"
PROJECT_LOGDIR = "file:./mlruns"

warnings.filterwarnings(
    "ignore"
)  # stop mlflow metric logger from going off all the time


def get_cuda_config() -> Dict[str, Any]:
    if torch.cuda.is_available():
        logger.info("cuda is available")
        return {
            "gpus": -1,
            "distributed_backend": "ddp",
        }

    else:
        logger.info("cuda not available")
        return {"distributed_backend": None}


def train():
    parser = ArgumentParser()

    parser.add_argument("--experiment-id", type=str, default="")
    experiment = get_experiment(parser.parse_args().experiment_id)
    logger.info(f"experiment name = {experiment.NAME}")

    mlflow_logger = MLFlowLogger(
        experiment_name=PROJECT_NAME,
        tags=experiment.TAGS or {},
        tracking_uri=PROJECT_LOGDIR,
    )

    parser = experiment.add_model_specific_args(parser)

    parser.add_argument(
        "--SEED",
        type=int,
        default=int(time.time()),
    )

    args = parser.parse_args()

    logger.info(f"Parsed args = {args}")
    pl.seed_everything(seed=args.SEED)

    mlflow_logger.log_hyperparams(args)

    trainer_callbacks = []
    if torch.cuda.is_available():
        trainer_callbacks.append(pl.callbacks.GPUStatsMonitor())

    trainer_callbacks.extend(
        [
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                mode="min",
            ),
            pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="step"),
        ]
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # filter because the other ddp processors don't have access to .name or .version
        dirpath="/".join(
            filter(
                None,
                [mlflow_logger.save_dir, mlflow_logger.name, mlflow_logger.version],
            )
        )
        + "/artifacts/weights",
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.2f}",
        verbose=False,
        save_last=True,
        save_top_k=5,
    )

    training_kwargs = get_cuda_config()

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=mlflow_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=trainer_callbacks,
        precision=16,
        **training_kwargs,
    )

    if experiment.TAGS.get("PYLIGHTNING_TUNE", False) is True:
        logger.info("PYLIGHTNING_TUNE is True")
        trainer.tune(experiment, experiment.data_module)

    experiment = experiment(**vars(args))
    trainer.fit(experiment, experiment.data_module)


if __name__ == "__main__":
    train()
