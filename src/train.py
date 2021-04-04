import time
from typing import Any, Dict

import click
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.experiments.experiments import get_experiment, get_experiment_names
import torch
import wilds

PROJECT_CRONJOB_NAME = "adrian-wilds-project"


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
@click.option("--dataset", type=str, default="default", help="<list of datasets>")
@click.option("--experiment-id", type=str, default="first-model")
def train(dataset: str, experiment_id: str):
    logger.info(f"dataset: {dataset}")
    logger.info(f"experiment_id: {experiment_id}")

    logger.info(wilds.benchmark_datasets)

    logger.info(get_experiment_names())

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
            pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="epoch"),
        ]
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="default",
        #{experiment.NAME}-{int(time.time())}
        tracking_uri="http://localhost:5000",
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
        max_epochs=5,
        callbacks=trainer_callbacks,
        **training_kwargs,
    )

    trainer.fit(experiment, experiment.data_module)


if __name__ == "__main__":
    train()
