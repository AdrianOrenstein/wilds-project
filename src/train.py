from typing import Any, Dict

import click
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.experiments.experiments import get_experiment
import torch

PROJECT_NAME = "adrian-wilds-project"
PROJECT_LOGDIR = "file:./mlruns"


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


@click.command()
@click.option("--experiment-id", type=str, default="")
def train(experiment_id: str):
    experiment = get_experiment(experiment_id)

    logger.info(f"experiment name = {experiment.NAME}")

    trainer_callbacks = []

    if torch.cuda.is_available():
        trainer_callbacks.append(pl.callbacks.GPUStatsMonitor())

    trainer_callbacks.extend(
        [
            pl.callbacks.EarlyStopping(
                monitor="2_val/val_loss",
                patience=75,
                mode="min",
            ),
            pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="step"),
        ]
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=PROJECT_NAME,
        tags=experiment.TAGS or {},
        tracking_uri=PROJECT_LOGDIR,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(mlflow_logger.save_dir),
        filename="{epoch}-{val_loss:.2f}",
        verbose=True,
        mode="min",
    )

    training_kwargs = get_cuda_config()

    trainer = pl.Trainer(
        logger=mlflow_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=trainer_callbacks,
        auto_scale_batch_size="power",
        precision=16,
        **training_kwargs,
        **experiment.TRAINING_KWARGS,
    )

    if experiment.TAGS["PYLIGHTNING_TUNE"] is True:
        logger.info("PYLIGHTNING_TUNE is True")
        trainer.tune(experiment, experiment.data_module)

    trainer.fit(experiment, experiment.data_module)


if __name__ == "__main__":
    train()
