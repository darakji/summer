import json
import os

input_file = '/home/phanim/harshitrawat/summer/MD_300_450K_strain_2_3.ipynb'
output_file = '/home/phanim/harshitrawat/summer/MD_300_450K_strain_2_3_clean.ipynb'

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # process the notebook
    if 'cells' in nb:
        for cell in nb['cells']:
            if 'outputs' in cell:
                cell['outputs'] = []
            if 'execution_count' in cell:
                cell['execution_count'] = None
                
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully created cleaned notebook at {output_file}")

except Exception as e:
    print(f"Error processing notebook: {e}")
