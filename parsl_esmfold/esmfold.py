"""Run ESMFold on a sequence to predict structure."""

from __future__ import annotations

import io
import re
from pathlib import Path

import torch
from Bio.PDB import PDBParser
from parsl_object_registry import clear_torch_cuda_memory_callback
from parsl_object_registry import register
from pydantic import BaseModel
from pydantic import Field

from parsl_esmfold.utils import exception_handler


class Sequence(BaseModel):
    """Amino acid sequence and description tag."""

    sequence: str = Field(
        ...,
        description='Amino acid sequence.',
    )
    tag: str = Field(
        ...,
        description='Sequence description tag (fasta header).',
    )


def read_fasta(fasta_file: str | Path) -> list[Sequence]:
    """Read fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile('^>', re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace('\n', '')
        for seq in non_parsed_seqs
        for line in seq.split('\n', 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag)
        for seq, tag in zip(lines[1::2], lines[::2])
    ]


def parse_plddt(pdb_file: str | io.StringIO) -> float:
    """Parse the pLDDT score from the structure file.

    Parameters
    ----------
    pdb_file : str
        The path to the PDB file.

    Returns
    -------
    float
        The mean pLDDT score of the structure.
    """
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure('id', pdb_file)

    # Collect B-factors
    b_factors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Get the B-factor of the atom
                    b_factors.append(atom.bfactor)

    # Calculate the mean B-factor
    plddt = sum(b_factors) / len(b_factors)

    return plddt


@register(shutdown_callback=clear_torch_cuda_memory_callback)
class EsmFold:
    """ESM-Fold model for protein structure prediction."""

    def __init__(self, torch_hub_dir: str | Path | None = None) -> None:
        """Initialize the ESM-Fold model.

        Parameters
        ----------
        torch_hub_dir : Optional[str]
            The path to the torch hub directory.
        """
        # Status message (should only be printed once per cold start)
        print('Loading ESMFold model into memory')

        # Configure the torch hub directory
        if torch_hub_dir is not None:
            torch.hub.set_dir(torch_hub_dir)

        # Load the model
        self.model = torch.hub.load('facebookresearch/esm:main', 'esmfold_v1')
        self.model.eval()

        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Optionally, uncomment to set a chunk size for axial attention.
        # This can help reduce memory. Lower sizes will have lower memory
        # requirements at the cost of increased speed.
        self.model.set_chunk_size(128)

    @exception_handler()
    @torch.no_grad()
    def run(self, sequence: str) -> tuple[str, float] | None:
        """Run the ESMFold model to predict structure.

        Parameters
        ----------
        sequence : str
            The sequence to fold.

        Returns
        -------
        tuple[str, float] | None
            str : The predicted structure in PDB format (as a string).
            float : The mean pLDDT score of the predicted structure.
            None : If the model fails.
        """
        # Get the predicted structure as a string containing PDB file contents
        structure = self.model.infer_pdb(sequence)

        # Extract the pLDDT score from the structure
        plddt = parse_plddt(io.StringIO(structure))

        return structure, plddt
