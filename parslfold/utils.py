"""Utility module."""

from __future__ import annotations

import functools
import io
import json
import re
import sys
from pathlib import Path
from typing import Callable
from typing import TypeVar

import yaml  # type: ignore[import-untyped]
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser
from pydantic import BaseModel as _BaseModel
from pydantic import Field

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar('T')
P = ParamSpec('P')


def exception_handler(
    default_return: T | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """Handle exceptions in a function by returning a `default_return` value.

    A decorator factory that returns a decorator formatted with the
    default_return that wraps a function.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                print(
                    f'{func.__name__} raised an exception: {e} '
                    f'On input {args}, {kwargs}\nReturning {default_return}',
                )
                return default_return

        return wrapper

    return decorator


class BaseModel(_BaseModel):
    """Provide an easy interface to read/write YAML files."""

    def dump_yaml(self, filename: str | Path) -> None:
        """Dump settings to a YAML file."""
        with open(filename, mode='w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], filename: str | Path) -> T:
        """Load settings from a YAML file."""
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


def batch_data(data: list[T], chunk_size: int) -> list[list[T]]:
    """Batch data into chunks of size chunk_size.

    Parameters
    ----------
    data : list[T]
        The data to batch.
    chunk_size : int
        The size of each batch.

    Returns
    -------
    list[list[T]]
        The batched data.
    """
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches


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
    """Read fasta file sequences and description tags into dataclass.

    Parameters
    ----------
    fasta_file : str | Path
        The path to the fasta file.

    Returns
    -------
    list[Sequence]
        A list of Sequence objects.
    """
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


def write_fasta(
    sequences: Sequence | list[Sequence],
    fasta_file: str | Path,
    mode: str = 'w',
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        f.write('\n'.join(f'>{seq.tag}\n{seq.sequence}' for seq in seqs))


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


def convert_cif_to_pdb(cif_file: str | Path, pdb_file: str | Path) -> None:
    """Convert a CIF file to a PDB file.

    Parameters
    ----------
    cif_file : str | Path
        Path to the input CIF file.
    pdb_file : str | Path
        Path to the output PDB file.

    Returns
    -------
    None
        The function saves the converted PDB file to the specified path.

    Examples
    --------
    >>> convert_cif_to_pdb("example.cif", "example.pdb")
    """
    # Parse the CIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure_name', str(cif_file))

    # Save as PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_file))
