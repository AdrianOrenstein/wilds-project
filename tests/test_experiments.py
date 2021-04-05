from src.train import (
    PROJECT_NAME,
)
from src.experiments.experiments import (
    get_experiment,
    get_experiment_names,
)
import torch


def test_experiment_name_lengths():
    experiment_names = get_experiment_names()

    for experiment_name in experiment_names:
        kubectl_job_name = "-".join(
            [PROJECT_NAME, experiment_name.replace("_", "-")]
        )
        assert (
            len(kubectl_job_name) <= 52
        ), f'"{kubectl_job_name}" must be less or equal to 52 runes'

