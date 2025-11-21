import json
from rdkit import Chem

dataset_names = ["N", "N_O", "N_minus_O"]

for dataset_name in dataset_names:
    PATH_TO_SDF = f"/home/cairne/WorkSpace/molgraphX_paper_scripts/Data/ibenchmark/Datasets/{dataset_name}_test_lbl.sdf"
    molecules = [mol for mol in Chem.SDMolSupplier(PATH_TO_SDF) if mol is not None]
    res_d = {}
    for i in molecules[:]:
        res_d[Chem.MolToSmiles(i)] = tuple(i.GetProp("lbls").split(","))

    with open(f"/home/cairne/WorkSpace/molgraphX_paper_scripts/GCNN_2D/Results/ibench_{dataset_name}_true_test.json", "w") as of:
        json.dump(res_d, of)
                