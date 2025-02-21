"""Basic example of using the Chai1 folding model."""

from __future__ import annotations

from pathlib import Path

from parslfold.chai1 import Chai1

sequence = 'MKVAVLGAAGGIGQALALLLKTQLPSGSELSLYDIAPVTPGVAVDLSHIPTAVKIKGFSGEDATPALEGADVVLISAGVARKPGMDRSDLFNVNAGIVKNLVQQVAKTCPKACIGIITNPVNTTVAIAAEVLKKAGVYDKNKLFGVTTLDIIRSNTFVAELKGKQPGEVEVPVIGGHSGVTILPLLSQVPGVSFTEQEVADLTKRIQNAGTEVVEAKAGGGSATLSMGQAAARFGLSLVRALQGEQGVVECAYVEGDGQYARFFSQPLLLGKNGVEERKSIGTLSAFEQNALEGMLDTLKKDIALGEEFVNK'  # noqa: E501

folding_model = Chai1(
    # The sequence type (e.g., protein, dna, rna, ligand).
    sequence_type='protein',
    # Whether to use ESM embeddings.
    use_esm_embeddings=True,
    # Number of trunk recycles.
    num_trunk_recycles=3,
    # The number of diffusion timesteps, by default 80
    # following recommendations from here:
    # https://github.com/chaidiscovery/chai-lab/issues/80
    num_diffn_timesteps=80,
    # Random seed.
    seed=42,
    # Device to use (cpu or cuda).
    device='cuda',
    # Path to the download directory.
    download_dir='examples/chai1',
)

# Set an output directory for each sequence
struct_output_dir = Path('examples/output_chai/')

# Create the output directory if it does not exist
struct_output_dir.mkdir(exist_ok=True, parents=True)

# Run the folding model
folding_model.run(sequence, output_dir=struct_output_dir)
