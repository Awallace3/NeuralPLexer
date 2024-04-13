import sys 

from rdkit import Chem
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt

filename = sys.argv[1]
with open(filename, 'rb') as f:
    rdkit_dict = pickle.load(f)
for key in rdkit_dict:
    for num, rdkit_obj in enumerate(rdkit_dict[key]):
        if num ==0:
            compare_obj = Chem.MolFromSmiles(key)
            compare_obj = Chem.AddHs(compare_obj)
            AllChem.EmbedMolecule(compare_obj, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(compare_obj)
            rmsd_list = []
        rmsd = Chem.rdMolAlign.GetBestRMS(rdkit_obj, compare_obj) 
        rmsd_list.append(rmsd)
    plt.hist(rmsd_list, bins=30)
    plt.xlabel('RMSD (Angstroms)')
    plt.title(key)
    plt.savefig(f'key'.png)
    break
