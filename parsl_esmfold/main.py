"""Main command-line interface for the ESMFold workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from parsl_esmfold.esmfold import read_fasta
from parsl_esmfold.esmfold import Sequence
from parsl_esmfold.parsl import ComputeConfigs
from parsl_esmfold.utils import BaseModel
from parsl_esmfold.utils import batch_data


class EsmFoldWorkflowConfig(BaseModel):
    """ESM-Fold workflow configuration."""

    fasta_path: Path = Field(
        ...,
        description='Path to the input fasta file or directory.',
    )
    output_dir: Path = Field(
        ...,
        description='Path to the output directory that writes a subdirectory '
        'for each sequence in the fasta file.',
    )
    torch_hub_dir: Path = Field(
        default=Path.home() / '.cache' / 'torch' / 'hub',
        description='Path to the torch hub directory.',
    )
    chunk_size: int = Field(
        default=1,
        description='Number of sequences to fold in each worker call.',
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
        if not self.torch_hub_dir.exists():
            raise FileNotFoundError(
                f'Torch hub directory not found: {self.torch_hub_dir}',
            )

        # Create the output directory if it does not exist
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Resolve the paths
        self.fasta_path = self.fasta_path.resolve()
        self.output_dir = self.output_dir.resolve()
        self.torch_hub_dir = self.torch_hub_dir.resolve()

        return self


def esmfold_worker(
    sequences: list[Sequence],
    torch_hub_dir: Path,
    output_dir: Path,
) -> None:
    """Run ESMFold on a list of sequences and write the results to disk.

    Parameters
    ----------
    sequences : list[Sequence]
        A list of sequences to fold.
    torch_hub_dir : Path
        The path to the torch hub directory.
    output_dir : Path
        The path to the output directory.
    """
    import json

    from parsl_esmfold.esmfold import EsmFold

    # Cache the model as module-level global to enable parsl warm starts
    esmfold = EsmFold(torch_hub_dir=torch_hub_dir)

    # Predict the structure for each sequence
    results = [esmfold.run(seq.sequence) for seq in sequences]

    # Write the results to disk
    for seq, result in zip(sequences, results):
        # Check if the result is None
        if result is None:
            print(f'Failed to fold sequence {seq.tag}')
            continue

        # Unpack the result
        structure, plddt = result

        # Write the structure to a PDB file
        with open(output_dir / f'{seq.tag}.pdb', 'w') as f:
            f.write(structure)

        # Write the pLDDT to a JSON file
        with open(output_dir / f'{seq.tag}.json', 'w') as f:
            json.dump({'plddt': plddt}, f, indent=4)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ESMFold workflow.')
    parser.add_argument(
        '--config_path',
        type=Path,
        help='Path to the configuration file.',
    )
    args = parser.parse_args()

    # Load the configuration
    config = EsmFoldWorkflowConfig.from_yaml(args.config_path)

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
            Sequence(sequence=seq.sequence, tag=f'{fasta_file.stem}_{i}')
            for i, seq in enumerate(seqs)
        ]

        sequences.extend(seqs)

    # Batch the sequences for parallel processing
    sequence_batches = batch_data(sequences, config.chunk_size)

    # Distribute the sequences across processes
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(esmfold_worker, sequence_batches))
