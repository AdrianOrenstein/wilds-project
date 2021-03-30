import click


@click.command()
@click.option("--dataset", type=str, default="default", help="<list of datasets>")
@click.option("--experiment-id", type=str, default="first-model")
def main(dataset: str, experiment_id: str):
    print(dataset, experiment_id)


if __name__ == "__main__":
    main()
