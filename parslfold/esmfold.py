"""Run ESMFold on a sequence to predict structure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch

# Check if we are on Aurora, if so import the XPU backend.
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex


from parsl_object_registry import clear_torch_cuda_memory_callback
from parsl_object_registry import register
from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput

from parslfold.utils import exception_handler


@register(shutdown_callback=clear_torch_cuda_memory_callback)
class EsmFold:
    """ESM-Fold model for protein structure prediction."""

    def __init__(
        self,
        device=Literal['cpu', 'cuda', 'xpu'],
        tokenizer: str = 'facebook/esmfold_v1',
        model: str = 'facebook/esmfold_v1',
        use_float16: bool = False,
        allow_tf32: bool = False,
        chunk_size: int | None = None,
    ) -> None:
        """Initialize the ESM-Fold model.

        Parameters
        ----------
        tokenizer : str
            The tokenizer to use.
        model : str
            The model to use.
        use_float16 : bool, optional
            Whether to use float16, by default False.
        allow_tf32 : bool, optional
            Whether to allow tf32, by default False.
        chunk_size : int | None, optional
            The chunk size for axial attention, by default None.
        platform : str, optional
            The GPU vendor this code will run on. Intel or NVIDIA.
        """
        # Status message (should only be printed once per cold start)
        print('Loading ESMFold model into memory')

        # Load the model and the tokenizer
        from transformers import AutoTokenizer
        from transformers import EsmForProteinFolding

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = EsmForProteinFolding.from_pretrained(model)
        self.model.eval()

        # Load the model to xpu, cuda, or cpu.
        self.device = torch.device(device)
        self.model.to(self.device)

        # Apply optimizations if requested in configs.
        if use_float16:
            self.model = self.model.half()
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Optionally, uncomment to set a chunk size for axial attention.
        # This can help reduce memory. Lower sizes will have lower memory
        # requirements at the cost of increased speed.
        if chunk_size is not None:
            self.model.set_chunk_size(chunk_size)

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
        # Tokenize the sequence with ESM-2 tokenizer.
        token_ids = self.tokenizer(
            [sequence],
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        token_ids = token_ids.to(self.device)
        esmfold_output = self.model(token_ids)

        # Convert the ESMFold output to a PDB string, plddt, and TM-Score.
        pdb_content, ptm, plddt = self._convert_to_pdb(esmfold_output)

        # Write the structure to a PDB file
        with open(Path(output_dir) / 'output.pdb', 'w') as f:
            f.write(pdb_content)

        # Write the pLDDT to a JSON file
        with open(Path(output_dir) / 'output.json', 'w') as f:
            json.dump({'plddt': plddt}, f, indent=4)
            json.dump({'pTM': ptm}, f, indent=4)

    @staticmethod
    def _convert_to_pdb(
        esmfold_output: EsmForProteinFoldingOutput,
    ) -> tuple[str, float, float]:
        """Convert the ESMFold output to a PDB string.

        Parameters
        ----------
        esmfold_output : EsmForProteinFoldingOutput
            The output of the ESMFold model.

        Returns
        -------
        tuple[str, float, float]
            str : The predicted structure in PDB format (as a string).
            float : The mean pLDDT score of the predicted structure.
            float : The TM-Score of the predicted structure.

        """
        from transformers.models.esm.openfold_utils.feats import (
            atom14_to_atom37,
        )
        from transformers.models.esm.openfold_utils.protein import (
            Protein as OFProtein,
        )
        from transformers.models.esm.openfold_utils.protein import to_pdb

        final_atom_positions = atom14_to_atom37(
            esmfold_output['positions'][-1],
            esmfold_output,
        )
        esmfold_output = {
            k: v.to('cpu').numpy() for k, v in esmfold_output.items()
        }
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = esmfold_output['atom37_atom_exists']
        pdbs = []
        # for each residue in the output object.
        for i in range(esmfold_output['aatype'].shape[0]):
            # Get the amino acid type, predicted positions, and mask.
            aa = esmfold_output['aatype'][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = esmfold_output['residue_index'][i] + 1
            # Create a protein object and convert it to a PDB string.
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=esmfold_output['plddt'][i],
                chain_index=esmfold_output['chain_index'][i]
                if 'chain_index' in esmfold_output
                else None,
            )
            pdbs.append(to_pdb(pred))

        ptm = esmfold_output['ptm'].item()
        plddt = esmfold_output['plddt'].mean().item()

        # concatenate the residue pdbs and return the result.
        return ''.join(pdbs), ptm, plddt
