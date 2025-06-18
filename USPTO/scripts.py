from rxnutils.chem.utils import remove_atom_mapping,remove_atom_mapping_template,remove_stereochemistry
import sys
sys.path.append("/root/retro_synthesis/template_analysis")
from tools.validation_format import check_format
from tools import load_data
from tools import draw_templates

uspto_data = load_data.load_USPTO_data()
Smiles_P_dict = uspto_data['Smiles_P_dict']
Smiles_R1_dict = uspto_data['Smiles_R1_dict']
Smiles_R2_dict = uspto_data['Smiles_R2_dict']
Reaction_template_dict = uspto_data['Reaction_template_dict']
Reaction_class_dict = uspto_data['Reaction_class_dict']

from drfp import DrfpEncoder
template_encodings = {}
for key,value in Reaction_template_dict.items():
    for k,v in value.items():
        if int(k) < 8:  # Only consider templates with index < 8
            template = v
            fps = DrfpEncoder.encode(template)[0]
            template_encodings[k] = fps
    if len(template_encodings) % 3000 == 0:
        print(f"Number of unique templates: {len(template_encodings)}")
print(f"Total number of unique templates: {len(template_encodings)}")

import pickle
with open('template_encodings.pkl', 'wb') as handle:
    pickle.dump(template_encodings, handle)