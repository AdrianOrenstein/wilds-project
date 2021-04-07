from argparse import ArgumentParser
import time
from typing import Any, Dict
import warnings

from loguru import logger
import mlflow
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


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path
        for f in mlflow.tracking.MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")


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

    pl.seed_everything(seed=args.SEED)

    logger.info(f"experiment args: {args}")
    mlflow_logger.log_hyperparams(args)

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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(mlflow_logger.save_dir),
        filename="{epoch}-{val_loss:.2f}",
        verbose=True,
        mode="min",
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

    # mlflow_logger.experiment.save_artifacts(
    #     run_id=mlflow_logger.experiment.experiment_id,
    #     local_dir='<model>.ckpt',
    #     artifact_path='models',
    # )

    # print_auto_logged_info(mlflow.get_run(run_id=mlflow_logger.run_id))


if __name__ == "__main__":
    train()
