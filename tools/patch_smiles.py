import pandas as pd
import glob
import os


def patch():
    print("Patching SMILES into enzyme_features_full.csv...")

    # 1. Load the broken CSV
    csv_path = 'data/enzyme_features_full.csv'
    if not os.path.exists(csv_path):
        print("CSV not found.")
        return
    df_features = pd.read_csv(csv_path)
    print(f"Loaded {len(df_features)} rows from CSV.")

    # 2. Load the source Parquet to get SMILES
    files = glob.glob("data/twosides_*_v18_clean.parquet")
    if not files:
        print("Parquet file not found.")
        return
    df_parquet = pd.read_parquet(files[0])

    # Build a Drug Name -> SMILES map
    drugs_a = df_parquet[['drug_a', 'smiles_a']].rename(columns={'drug_a': 'drug_name', 'smiles_a': 'smiles'})
    drugs_b = df_parquet[['drug_b', 'smiles_b']].rename(columns={'drug_b': 'drug_name', 'smiles_b': 'smiles'})
    df_map = pd.concat([drugs_a, drugs_b]).drop_duplicates(subset='drug_name')

    # 3. Merge
    # We drop 'smiles' from features if it exists (to avoid collision) and merge fresh
    if 'smiles' in df_features.columns:
        df_features = df_features.drop(columns=['smiles'])

    df_fixed = pd.merge(df_features, df_map, on='drug_name', how='left')

    # 4. Save
    df_fixed.to_csv(csv_path, index=False)
    print(f"Success! Saved patched CSV with {len(df_fixed)} rows and columns: {list(df_fixed.columns)[:3]}...")


if __name__ == "__main__":
    patch()
