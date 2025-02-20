"""Basic example of using the ESMFold folding model."""

from __future__ import annotations

from pathlib import Path

from parslfold.esmfold import EsmFold

sequence = 'MKVAVLGAAGGIGQALALLLKTQLPSGSELSLYDIAPVTPGVAVDLSHIPTAVKIKGFSGEDATPALEGADVVLISAGVARKPGMDRSDLFNVNAGIVKNLVQQVAKTCPKACIGIITNPVNTTVAIAAEVLKKAGVYDKNKLFGVTTLDIIRSNTFVAELKGKQPGEVEVPVIGGHSGVTILPLLSQVPGVSFTEQEVADLTKRIQNAGTEVVEAKAGGGSATLSMGQAAARFGLSLVRALQGEQGVVECAYVEGDGQYARFFSQPLLLGKNGVEERKSIGTLSAFEQNALEGMLDTLKKDIALGEEFVNK'  # noqa: E501

folding_model = EsmFold(
    # The sequence type (e.g., protein, dna, rna, ligand).
    tokenizer='facebook/esmfold_v1',
    model='facebook/esmfold_v1',
    use_float16=False,
    allow_tf32=False,
    chunk_size=None,
)

# Set an output directory for each sequence
struct_output_dir = Path('/homes/ogokdemir/parslfold/examples/output')

# Create the output directory if it does not exist
struct_output_dir.mkdir(exist_ok=True, parents=True)

# Run the folding model
folding_model.run(sequence, output_dir=struct_output_dir)
