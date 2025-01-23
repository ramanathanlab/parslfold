# parslfold
Fold proteins in parallel using [Parsl](https://parsl-project.org/).

Supported folding methods:
- [Chai-1](https://github.com/chaidiscovery/chai-lab)
- [ESMFold](https://github.com/facebookresearch/esm?tab=readme-ov-file#esmfold) (WIP)

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:ramanathanlab/parslfold.git
cd parslfold
pip install -U pip setuptools wheel
pip install -e .
```

### Installation on Polaris

To install the package on Polaris@ALCF, run the following commands before the pip install command:
```bash
module use /soft/modulefiles; module load conda
```

## Usage

To fold a set of proteins, run the following command (see example YAML config for details):
```bash
nohup python -m parslfold.main --config examples/workstation_config.yaml &> nohup.log &
```

The output folder structure will look like this:
```
examples/output/
├── config.yaml
├── parsl
│   └── 000
│       ├── htex
│       │   ├── block-0
│       │   │   └── 082881fe477f
│       │   │       ├── manager.log
│       │   │       ├── worker_0.log
│       │   │       └── worker_1.log
│       │   └── interchange.log
│       ├── parsl.log
│       └── submit_scripts
│           ├── parsl.htex.block-0.1737608324.8257468.sh
│           ├── parsl.htex.block-0.1737608324.8257468.sh.ec
│           ├── parsl.htex.block-0.1737608324.8257468.sh.err
│           └── parsl.htex.block-0.1737608324.8257468.sh.out
└── structures
    ├── uniprotkb_accession_A0LFF8_OR_accession_2024_12_19_seq_0
    │   ├── input.fasta
    │   ├── pred.model_idx_0.pdb
    │   └── scores.model_idx_0.npz
    └── uniprotkb_accession_A0LFF8_OR_accession_2024_12_19_seq_1
        ├── input.fasta
        ├── pred.model_idx_4.pdb
        └── scores.model_idx_4.npz
```

- `config.yaml`: The configuration file used to run the folding.
- `parsl/`: The Parsl logs and submit scripts (containing stdout and stderr).
- `structures/`: The folded protein structures.
    - `input.fasta`: The input sequence used for folding.
    - `pred.model_idx_X.pdb`: The highest confidence folded protein structure.
    - `scores.model_idx_X.npz`: The scores for the folded protein structure.

### Notes
- We only keep the highest confidence folded protein structure and its scores.
- The subdirectories within `structures/` are named based on the input sequence fasta file name and the index of the sequence in the file (e.g., `<fasta-name>_seq_X`).
- See `examples/chai1_example.py` for a quick example of how to fold a protein using the Chai-1 method in our package. This is the same core functionality as the main script, but without the parallelism provided by Parsl.


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
