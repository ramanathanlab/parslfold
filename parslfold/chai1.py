"""Run Chai-1 on a sequence to predict structure."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import torch
from parsl_object_registry import clear_torch_cuda_memory_callback
from parsl_object_registry import register

from parslfold.utils import exception_handler
from parslfold.utils import Sequence
from parslfold.utils import write_fasta


@register(shutdown_callback=clear_torch_cuda_memory_callback)
class Chai1:
    """Chai-1 model for protein structure prediction."""

    def __init__(
        self,
        sequence_type: str = 'protein',
        use_esm_embeddings: bool = True,
        num_trunk_recycles: int = 3,
        num_diffn_timesteps: int = 200,
        seed: int = 42,
        device: str = 'cuda',
        download_dir: str | Path | None = None,
        tmp_dir: str | Path = Path('/dev/shm'),
    ) -> None:
        """Initialize the Chai-1 model.

        Parameters
        ----------
        sequence_type : str
            The sequence type (e.g., 'protein', 'dna', 'rna', 'ligand').
        use_esm_embeddings : bool
            Whether to use ESM embeddings.
        num_trunk_recycles : int
            The number of trunk recycles.
        num_diffn_timesteps : int
            The number of diffn timesteps.
        seed : int
            The random seed.
        device : str
            The device to use.
        download_dir : str | Path | None
            The path to the download directory.
        tmp_dir : str | Path
            The temporary directory.
        """
        self.sequence_type = sequence_type
        self.use_esm_embeddings = use_esm_embeddings
        self.num_trunk_recycles = num_trunk_recycles
        self.num_diffn_timesteps = num_diffn_timesteps
        self.seed = seed
        self.device = device
        self.tmp_dir = Path(tmp_dir)

        # Set the download directory if provided
        if download_dir is not None:
            os.environ['CHAI_DOWNLOADS_DIR'] = str(download_dir)

    @exception_handler()
    @torch.no_grad()
    def run(self, sequence: str, output_dir: str | Path) -> None:
        """Run the Chai-1 model to predict structure.

        Parameters
        ----------
        sequence : str
            The sequence to fold.
        output_dir : str | Path
            The output directory.
        """
        from chai_lab.chai1 import run_inference

        # Create a temporary directory to write intermediate files
        tmp_dir = self.tmp_dir / str(uuid.uuid4())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Chai-1 requires a fasta file as input with specific header
        seq = Sequence(
            sequence=sequence,
            tag=f'{self.sequence_type}|name=seq0',
        )
        tmp_fasta = tmp_dir / 'seq.fasta'
        write_fasta(seq, tmp_fasta)

        # Run the folding model
        candidates = run_inference(
            fasta_file=tmp_fasta,
            output_dir=tmp_dir / 'output',
            num_trunk_recycles=self.num_trunk_recycles,
            num_diffn_timesteps=self.num_diffn_timesteps,
            seed=self.seed,
            device=self.device,
            use_esm_embeddings=self.use_esm_embeddings,
        )

        # Find the best candidate
        # The cif files are of the form pred.model_idx_0.cif
        cif_paths = candidates.cif_paths
        scores = [x.aggregate_score.item() for x in candidates.ranking_data]
        best_idx = np.argmax(scores)
        best_cif = cif_paths[best_idx]

        # We also want to get scores file with format scores.model_idx_0.npz
        scores_path = best_cif.with_name(
            best_cif.name.replace('pred', 'scores').replace('.cif', '.npz'),
        )

        # TODO: Convert the cif file to PDB format

        # Copy the best candidate to the output directory
        best_cif.rename(Path(output_dir) / best_cif.name)
        scores_path.rename(Path(output_dir) / scores_path.name)

        # Clean up the temporary fasta file
        tmp_dir.unlink()
