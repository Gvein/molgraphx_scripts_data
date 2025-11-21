import sys
import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem
from tqdm import tqdm
from torch import Tensor, nan, nan_to_num

ROOT_DIR = "/home/cairne/WorkSpace/molgraphX_paper_scripts/GCNN_2D"
sys.path.append(ROOT_DIR)
from Source.explainers.utils import visualize

from Source.explainers import ExplainableModel
from Source.models.GCNN.featurizers import DGLFeaturizer
from Source.models.GCNN.model import GCNN
from Source.trainer import ModelShell
from Source.explainers import (ExplainableModel, get_molgraphX_scores,
                               get_submoleculeX_scores, get_subgraphX_subgraphs, 
                               draw_subgraphX_best_subgraph, get_subgraphX_scores)
from Source.explainers.subgraphX.utils import calculate_atom_scores_multiple_coalitions

DEVICE = "cpu"


def get_model_and_featurizer(path_to_model: str) -> ():
    MODEL = ExplainableModel(
        ModelShell(
            GCNN,
            path_to_model,
            device=DEVICE
        ))

    FEATURIZER = DGLFeaturizer(
        add_self_loop=False,
        node_featurizer=CanonicalAtomFeaturizer(),
        require_edge_features=False,
        canonical_atom_order=False,
    )

    return MODEL, FEATURIZER

def get_ig_scores(mol, model, featurizer):
    graph = featurizer.featurize(mol)

    graph = graph.to(DEVICE)
    model.to(DEVICE)
    model.eval()
    input_tensor = graph.x.clone().detach().requires_grad_(True)

    class CaptumModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            g = graph.clone()
            g.x = x
            return self.model(g)

    ig = IntegratedGradients(CaptumModelWrapper(model))
    attributions, approximation_error = ig.attribute(
        input_tensor,
        target=0,
        n_steps=50,
        method='gausslegendre',
        return_convergence_delta=True,
    )
    atom_importance = attributions.detach().cpu().numpy().sum(axis=1).tolist()
    
    return atom_importance

def get_molgraphx_scores(mol, model, featurizer, mode="regression", min_atoms=5) -> ():
    molgraphX_kwargs = {
        "min_atoms": min_atoms,
    }


    molgraphX_scores = get_molgraphX_scores(
        mol,
        featurizer=featurizer,
        explainable_model=model,
        # device=DEVICE,
        explainer_kwargs=molgraphX_kwargs,
        # is_sym=True,
        target=0,
        mode=mode
    )

    return np.array(nan_to_num(molgraphX_scores)).tolist()


def get_molgraphx_scores_raw(mol, model, featurizer, mode="regression"):
    # predictor returns a constant scalar
    def predictor(mols, model):
        return torch.tensor([10.0 for _ in mols])

    explainer = AtomsExplainer(predictor, min_atoms=mol.GetNumAtoms())
    scores = explainer(mol)


def get_subgX_scores(mol, model, featurizer, mode="regression", min_atoms=3) -> ():
    subgraphX_kwargs = {
        "mode": "regression",
        "device": DEVICE,
        "explain_graph": True,  # verbose: True,
        "rollout": 20,  # Number of iteration to get the prediction (MCTS hyperparameter)
        "min_atoms": 1,
        "c_puct": 10.0,  # The hyperparameter which encourages the exploration (MCTS hyperparameter)
        "sample_num": None,
        "reward_method": "l_shapley",  # one of ["gnn_score", "mc_shapley", "l_shapley", "mc_l_shapley", "nc_mc_l_shapley"]
        "subgraph_building_method": "zero_filling",  # one of ["zero_filling", "split"]
    }

    graph = featurizer.featurize(mol)

    subgraphs = get_subgraphX_scores(
        mol,
        featurizer=featurizer,
        explainable_model=model,
        device=DEVICE,
        subgraphX_kwargs=subgraphX_kwargs,
        target_ids=(0,),)
    atom_scores, important_atoms, stats = calculate_atom_scores_multiple_coalitions(subgraphs[0], 
                                                                                    num_atoms=mol.GetNumAtoms(),
                                                                                    method="binary_majority")
    return atom_scores
