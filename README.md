<div align="center">    
 
# Wilds-project

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->

<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->

<!--
Conference
-->
</div>
 
## Description   
What it does

## How to run an interactive terminal

First, download docker.

Second, run for an interactive terminal

```bash
make run
```

## How to run an experiment

First, download docker.

Second, run for an interactive terminal

```bash
make run -- python src/train.py
    --experiment-id 'test_model'
```

## How to make an experiment

1. See example test experiment at `src/experiments/test_model.py`
2. Inside of `src/experiments/experiments.py` import the new experiment
3. Run `make run -- python src/train.py --experiment-id '<experiment_name>'`

## Linting

```bash
make lint
```

## Tests

```bash
make test
```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
