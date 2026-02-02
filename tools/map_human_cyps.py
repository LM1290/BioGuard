import json
import os
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm


def fetch_human_cyps():
    print("Querying ChEMBL for human Cytochrome P450 targets...")
    target = new_client.target


    res = target.filter(target_type='SINGLE PROTEIN') \
        .filter(organism='Homo sapiens') \
        .filter(pref_name__icontains='Cytochrome P450')

    cyp_map = {}
    print(f"Found {len(res)} potential targets. Filtering...")

    for t in tqdm(res):
        chembl_id = t['target_chembl_id']
        name = t['pref_name']


        parts = name.split(" ")
        clean_name = None

        if len(parts) > 2 and parts[0] == "Cytochrome" and parts[1] == "P450":
            suffix = parts[-1]
            if suffix[0].isdigit():  # e.g., 1A2, 2D6
                clean_name = f"CYP{suffix}"

        if clean_name:
            cyp_map[chembl_id] = clean_name

    print(f"Mapped {len(cyp_map)} unique CYP targets.")

    os.makedirs('data', exist_ok=True)
    with open('data/cyp_target_map.json', 'w') as f:
        json.dump(cyp_map, f, indent=2)

    print("Saved to data/cyp_target_map.json")


if __name__ == "__main__":
    fetch_human_cyps()