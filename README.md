# parsl_esmfold
Implementation of [ESMFold](https://www.science.org/doi/10.1126/science.ade2574) using [Parsl](https://parsl-project.org/).

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:ramanathanlab/parsl_esmfold.git
cd parsl_esmfold
pip install -U pip setuptools wheel
pip install -e .
```

### Installation on Polaris

To install the package on Polaris@ALCF, run the following commands:
```bash
module use /soft/modulefiles; module load conda
```

Follow the full installation instructions above, and install torch via:
```bash
pip install torch
```

## Usage
TODO

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
