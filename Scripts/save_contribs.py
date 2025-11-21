import os
import pandas as pd
import metrics as mt
import os
import json
from rdkit import Chem
import numpy as np
import torch.nn as nn
from captum.attr import IntegratedGradients
from dgllife.utils import CanonicalAtomFeaturizer
from explainers_callers import (get_model_and_featurizer, get_ig_scores, get_subgX_scores,
                                get_molgraphx_scores, DEVICE)   
from Source.explainers import visualize
from tqdm import tqdm
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from torch import Tensor, nan, nan_to_num
from Source.explainers.subgraphX.utils import calculate_atom_scores_multiple_coalitions

dataset_names = [
                "N", 
                "N_O",
                "N_minus_O"
                ]
method_d = {
        #    "ig": get_ig_scores, 
        #    "molgx": get_molgraphx_scores,
           "subgx": get_subgX_scores
            }

for method in method_d.keys():
    for dataset_name in dataset_names:
        MODEL_FOLDER = f"/home/cairne/WorkSpace/molgraphX_paper_scripts/GCNN_2D/Output/ibench_{dataset_name}"
        PATH_TO_SDF = f"/home/cairne/WorkSpace/molgraphX_paper_scripts/Data/ibenchmark/Datasets/{dataset_name}_test_lbl.sdf"

        model, featurizer = get_model_and_featurizer(MODEL_FOLDER)
        molecules = [mol for mol in Chem.SDMolSupplier(PATH_TO_SDF) if mol is not None]
        molecules.sort(key=lambda mol: mol.GetNumAtoms())

        def normalize(x: list[float]) -> list[float]:
            numpy_x = np.array(x)
            norm_x = (numpy_x - numpy_x.mean()) / (numpy_x.std() + 1e-10)
            return norm_x.tolist()

        def wrapper(mol):
            return method_d[method](mol, model, featurizer)

        res_d = {}
        for i in tqdm(molecules[:100]):
            results = wrapper(i)
            res_d[Chem.MolToSmiles(i)] = tuple(results)

        with open(f"/home/cairne/WorkSpace/molgraphX_paper_scripts/GCNN_2D/Results/ibench_{dataset_name}_{method}_test.json", "w") as of:
            json.dump(res_d, of)
                