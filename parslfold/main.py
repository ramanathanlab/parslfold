"""Main command-line interface for the ESMFold workflow."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from parslfold.parsl import ComputeConfigs
from parslfold.utils import BaseModel
from parslfold.utils import batch_data
from parslfold.utils import read_fasta
from parslfold.utils import Sequence


class EsmFoldConfig(BaseModel):
    """ESM-Fold configuration."""

    torch_hub_dir: Path = Field(
        default=Path.home() / '.cache' / 'torch' / 'hub',
        description='Path to the torch hub directory.',
    )

    @model_validator(mode='after')
    def _validate_paths(self) -> Self:
        if not self.torch_hub_dir.exists():
            raise FileNotFoundError(
                f'Torch hub directory not found: {self.torch_hub_dir}',
            )

        # Resolve the path
        self.torch_hub_dir = self.torch_hub_dir.resolve()

        return self


# TODO: Fill in the configs once it's done.
class ESMFoldHFConfig(BaseModel):
    """TODO: Fill in the configs once it's done."""

    pass


class Chai1Config(BaseModel):
    """Chai-1 configuration."""

    sequence_type: str = Field(
        default='protein',
        description='The sequence type (e.g., protein, dna, rna, ligand).',
    )
    use_esm_embeddings: bool = Field(
        default=True,
        description='Whether to use ESM embeddings.',
    )
    num_trunk_recycles: int = Field(
        default=3,
        description='Number of trunk recycles.',
    )
    num_diffn_timesteps: int = Field(
        default=80,
        description='The number of diffusion timesteps, by default 80 '
        'following recommendations from here: '
        'https://github.com/chaidiscovery/chai-lab/issues/80',
    )
    seed: int = Field(
        default=42,
        description='Random seed.',
    )
    device: str = Field(
        default='cuda',
        description='Device to use (cpu or cuda).',
    )
    download_dir: Path | None = Field(
        default=None,
        description='Path to the download directory.',
    )
    tmp_dir: str | Path = Field(
        default=Path('/dev/shm'),
        description='Temporary directory for storing intermediate files.',
    )

    @model_validator(mode='after')
    def _validate_paths(self) -> Self:
        # Resolve the path
        if self.download_dir is not None:
            self.download_dir = self.download_dir.resolve()

        return self


class ParslFoldWorkflowConfig(BaseModel):
    """Parslfold workflow configuration."""

    fasta_path: Path = Field(
        ...,
        description='Path to the input fasta file or directory.',
    )
    output_dir: Path = Field(
        ...,
        description='Path to the output directory that writes a subdirectory '
        'for each sequence in the fasta file.',
    )
    chunk_size: int = Field(
        default=1,
        description='Number of sequences to fold in each worker call.',
    )
    folding_config: EsmFoldConfig | Chai1Config = Field(
        ...,
        description='Configuration for the folding model [ESM-Fold, Chai-1].',
    )
    compute_config: ComputeConfigs = Field(
        ...,
        description='A Parsl compute configuration for the ESM-Fold workflow.',
    )

    @model_validator(mode='after')
    def _validate_paths(self) -> Self:
        # Check if the paths exist
        if not self.fasta_path.exists():
            raise FileNotFoundError(f'Fasta path not found: {self.fasta_path}')

        # Create the output directory if it does not exist
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Resolve the paths
        self.fasta_path = self.fasta_path.resolve()
        self.output_dir = self.output_dir.resolve()

        return self


def parslfold_worker(
    sequences: list[Sequence],
    output_dir: Path,
    folding_method: str = 'chai1',
    **folding_kwargs: dict[str, Any],
) -> None:
    """Fold a list of sequences and write the results to disk.

    Parameters
    ----------
    sequences : list[Sequence]
        A list of sequences to fold.
    output_dir : Path
        The path to the output directory.
    folding_method : str
        The folding method to use, either 'chai1' or 'esmfold'.
    folding_kwargs : dict[str, Any]
        Additional keyword arguments to pass to the folding function.
    """
    from parslfold.utils import write_fasta

    # Create the output directory if it does not exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Instantiation caches the model as module-level
    # global to enable parsl warm starts
    if folding_method == 'chai1':
        from parslfold.chai1 import Chai1

        folding_model = Chai1(**folding_kwargs)  # type: ignore
    elif folding_method == 'esmfold':
        from parslfold.esmfold import EsmFold

        folding_model = EsmFold(**folding_kwargs)  # type: ignore

    # Fold each sequence and write the results to disk
    # the directories are named as {fasta_file.stem}_seq_{i}
    for seq in sequences:
        # Set an output directory for each sequence
        struct_output_dir = output_dir / seq.tag

        # Create the output directory if it does not exist
        struct_output_dir.mkdir(exist_ok=True, parents=True)

        # Run the folding model
        folding_model.run(seq.sequence, output_dir=struct_output_dir)

        # Write the sequence to a file for logging
        write_fasta(seq, struct_output_dir / 'input.fasta')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Parslfold workflow.')
    parser.add_argument(
        '--config_path',
        type=Path,
        help='Path to the configuration file.',
    )
    args = parser.parse_args()

    # Load the configuration
    config = ParslFoldWorkflowConfig.from_yaml(args.config_path)

    # Log the configuration
    config.dump_yaml(config.output_dir / 'config.yaml')

    # Set the parsl compute settings
    parsl_config = config.compute_config.get_parsl_config(
        config.output_dir / 'parsl',
    )

    # Check if the fasta_path is a directory, then gather all fasta files
    if config.fasta_path.is_dir():
        fasta_files = list(config.fasta_path.glob('*.fasta'))
        if not fasta_files:
            raise ValueError(f'No fasta files found in {config.fasta_path}')
    else:
        # Single fasta file
        fasta_files = [config.fasta_path]

    # Read all the sequences to memory
    sequences = []
    for fasta_file in fasta_files:
        # Read the file
        seqs = read_fasta(fasta_file)

        # Add a tag to each sequence containing the fasta file name
        # and the sequence index (to be used as a way to name the output files)
        seqs = [
            Sequence(sequence=seq.sequence, tag=f'{fasta_file.stem}_seq_{i}')
            for i, seq in enumerate(seqs)
        ]

        sequences.extend(seqs)

    # Batch the sequences for parallel processing
    sequence_batches = batch_data(sequences, config.chunk_size)

    # Decide which folding method to use
    if isinstance(config.folding_config, EsmFoldConfig):
        folding_method = 'esmfold'
    else:
        folding_method = 'chai1'

    # Define the worker function with fixed arguments
    worker_fn = functools.partial(
        parslfold_worker,
        output_dir=config.output_dir / 'structures',
        folding_method=folding_method,
        **config.folding_config.model_dump(),
    )

    # Distribute the sequences across processes
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(worker_fn, sequence_batches))
