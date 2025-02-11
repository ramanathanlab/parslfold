"""Run ESMFold on a sequence to predict structure."""

from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from parsl_object_registry import clear_torch_cuda_memory_callback
from parsl_object_registry import register

from parslfold.utils import exception_handler
from parslfold.utils import parse_plddt


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
    def run(
        self,
        sequence: str,
        output_dir: str | Path,
    ) -> tuple[str, float] | None:
        """Run the ESMFold model to predict structure.

        Parameters
        ----------
        sequence : str
            The sequence to fold.
        output_dir : str | Path
            The path to the output directory.

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

        # Write the structure to a PDB file
        with open(Path(output_dir) / 'output.pdb', 'w') as f:
            f.write(structure)

        # Write the pLDDT to a JSON file
        with open(Path(output_dir) / 'output.json', 'w') as f:
            json.dump({'plddt': plddt}, f, indent=4)

        return structure, plddt
