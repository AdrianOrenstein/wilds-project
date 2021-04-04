import argparse
import concurrent.futures
from itertools import repeat
import os
import sys
from typing import List

import click
import wilds


def download_dataset(dataset, root_dir):
    """
    Downloads the specified dataset
    """

    print(f"=== {dataset} ===")

    return wilds.get_dataset(dataset=dataset, root_dir=root_dir, download=True)


def main():
    """
    Downloads the latest versions of all specified datasets,
    if they do not already exist.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        required=False,
        default="data",
        help="The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Specify a space-separated list of dataset names to download. If left unspecified, the script will download all of the official benchmark datasets. Available choices are {wilds.supported_datasets}.",
    )
    config = parser.parse_args()

    datasets = config.datasets
    root_dir = config.root_dir

    if not datasets:
        datasets = wilds.benchmark_datasets

    invalid_datasets = [
        dataset for dataset in datasets if dataset not in wilds.supported_datasets
    ]

    if invalid_datasets:
        raise ValueError(
            f"{invalid_datasets} not recognized; must be one of {wilds.supported_datasets}."
        )

    print(f"Downloading the following datasets: {datasets}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = executor.map(download_dataset, datasets, repeat(root_dir))

        for result in futures:
            print(result)


if __name__ == "__main__":
    main()
