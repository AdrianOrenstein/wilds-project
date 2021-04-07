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
[WILDs](https://wilds.stanford.edu/) is a benchmark of in-the-wild distribution shifts spanning diverse data modalities and applications, from tumor identification to wildlife monitoring to poverty mapping.

This repo contains my infrastructure to maintain experiments on the dataset.

## How to run an interactive terminal in my environment

First, download docker.

Second, run for an interactive terminal

```bash
make run
```

## How to run an experiment

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

## Citation

```
@article{wilds-project,
  title={Wilds project},
  author={Adrian Orenstein},
  journal={GitHub. Note: https://github.com/AdrianOrenstein/wilds-project},
  year={2021}
}
```

#### push to public

`git push public public:master`
