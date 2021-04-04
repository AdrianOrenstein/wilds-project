import click
from loguru import logger
import torch
import wilds


@click.command()
@click.option("--dataset", type=str, default="default", help="<list of datasets>")
@click.option("--experiment-id", type=str, default="first-model")
def main(dataset: str, experiment_id: str):
    logger.info(f"dataset: {dataset}")
    logger.info(f"experiment_id: {experiment_id}")

    logger.info(wilds.benchmark_datasets)


if __name__ == "__main__":
    main()
