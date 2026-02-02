import pandas as pd
import json
import time
import os
import glob
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from tqdm import tqdm


def get_chembl_id(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        inchikey = Chem.MolToInchiKey(mol)
        res = new_client.molecule.filter(molecule_structures__standard_inchi_key=inchikey).only('molecule_chembl_id')
        if res: return res[0]['molecule_chembl_id']
    except:
        return None
    return None


def load_unique_drugs_from_parquet():
    """Finds the latest parquet cache and extracts unique drugs."""
    # Look for any v16 clean parquet file
    files = glob.glob("data/twosides_*_v16_clean.parquet")
    if not files:
        raise FileNotFoundError("No cached parquet file found in data/. Run data_loader.py first.")

    target_file = files[0]  # Pick the first one found
    print(f"Loading drugs from {target_file}...")

    df = pd.read_parquet(target_file)

    # Stack drug_a and drug_b to get unique list
    drugs_a = df[['drug_a', 'smiles_a']].rename(columns={'drug_a': 'drug_name', 'smiles_a': 'smiles'})
    drugs_b = df[['drug_b', 'smiles_b']].rename(columns={'drug_b': 'drug_name', 'smiles_b': 'smiles'})

    unique_df = pd.concat([drugs_a, drugs_b]).drop_duplicates(subset='drug_name').reset_index(drop=True)
    print(f"Found {len(unique_df)} unique drugs in dataset.")
    return unique_df


def main():
    if not os.path.exists('data/cyp_target_map.json'):
        print("Please run tools/map_human_cyps.py first!")
        return

    with open('data/cyp_target_map.json', 'r') as f:
        target_map = json.load(f)

    valid_targets = set(target_map.keys())
    all_cyps = sorted(list(set(target_map.values())))

    # Schema: Drug | CYP1A1_sub | CYP1A1_inh ...
    columns = ['drug_name']
    for cyp in all_cyps:
        columns.append(f"{cyp}_sub")
        columns.append(f"{cyp}_inh")

    df_drugs = load_unique_drugs_from_parquet()
    results = []

    print(f"Profiling against {len(all_cyps)} enzymes...")

    for idx, row in tqdm(df_drugs.iterrows(), total=len(df_drugs)):
        name = row['drug_name']
        smiles = row['smiles']
        cid = get_chembl_id(smiles)

        row_data = {col: 0 for col in columns}
        row_data['drug_name'] = name

        if cid:
            mechs = new_client.mechanism.filter(molecule_chembl_id=cid)
            for m in mechs:
                tid = m.get('target_chembl_id')
                if tid in valid_targets:
                    cyp_name = target_map[tid]
                    action = m.get('action_type', '').lower()

                    if 'inhibitor' in action:
                        row_data[f"{cyp_name}_inh"] = 1
                    if 'substrate' in action or 'metabolism' in action:
                        row_data[f"{cyp_name}_sub"] = 1

        results.append(row_data)
        if idx % 10 == 0: time.sleep(0.1)

    out_df = pd.DataFrame(results, columns=columns)
    out_df.to_csv('data/enzyme_features_full.csv', index=False)
    print("Saved comprehensive profile to data/enzyme_features_full.csv")


if __name__ == "__main__":
    main()