# ScandiDPR

Dense Passage Retrieval models for the Scandinavian languages.

______________________________________________________________________
[![PyPI Status](https://badge.fury.io/py/scandi_dpr.svg)](https://pypi.org/project/scandi_dpr/)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://saattrupdan.github.io/ScandiDPR/scandi_dpr.html)
[![License](https://img.shields.io/github/license/saattrupdan/ScandiDPR)](https://github.com/saattrupdan/ScandiDPR/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/ScandiDPR)](https://github.com/saattrupdan/ScandiDPR/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/saattrupdan/ScandiDPR/tree/main/tests)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/saattrupdan/ScandiDPR/blob/main/CODE_OF_CONDUCT.md)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```
$ poetry add <package-name>
```

### Get an overview of the available commands

Simply write `make` to display a list of the commands available. This includes the
above-mentioned `make install` command, as well as building and viewing documentation,
publishing the code as a package and more.


## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project


## Project structure
```
.
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   └── hydra
│       └── job_logging
│           └── custom.yaml
├── makefile
├── models
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scandi_dpr
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── data_collator.py
│   │   ├── evaluate.py
│   │   ├── loss.py
│   │   ├── model.py
│   │   ├── tokenization.py
│   │   ├── train.py
│   │   └── utils.py
│   └── scripts
│       ├── fix_dot_env_file.py
│       ├── train_model.py
│       └── versioning.py
└── tests
    ├── __init__.py
    └── test_dummy.py
```
