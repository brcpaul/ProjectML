from utils import flatten_dict_columns
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def parse_cytogenetics(df, col="CYTOGENETICS"):
    """
    Fonction pour extraire des features numériques à partir de la nomenclature cytogénétique.
    """    

    df["num_anomalies"] = df[col].apply(lambda x: len(re.findall(r'\b(del|t|inv|ins|dup|der|add|i)\b', str(x))))
    df["has_translocation"] = df[col].apply(lambda x: 1 if "t(" in str(x) else 0)
    df["has_deletion"] = df[col].apply(lambda x: 1 if "del(" in str(x) else 0)
    df["has_monosomy"] = df[col].apply(lambda x: 1 if re.search(r'\b-\d+\b', str(x)) else 0)
    df["has_complex_karyotype"] = df["num_anomalies"].apply(lambda x: 1 if x >= 3 else 0)
    
    # Comptage des cellules affectées et calcul de la clonalité
    def extract_clonal_percentage(cytogenetics_str):
        matches = re.findall(r'\[(\d+)\]', str(cytogenetics_str))
        if matches:
            total_cells = sum(map(int, matches))
            if total_cells > 0:
                return max(map(int, matches)) / total_cells
        return 0
    
    df["clonal_percentage"] = df[col].apply(extract_clonal_percentage)
    
    return df.drop(columns=["CYTOGENETICS"])  # On garde uniquement les features numériques

def process_molecular_data(maf_df:pd.DataFrame, x_train_molecular:pd.DataFrame, y_train_molecular:pd.DataFrame):
    train = x_train_molecular.merge(y_train_molecular, on="ID")
    maf_df = maf_df.copy()
    
    maf_df["REFALT"] = maf_df["REF"] +" "+ maf_df["ALT"]
    train["REFALT"] = train["REF"] +" "+ train["ALT"]
    
    rounding_factor = 1
    # Mapping de la position dans le chromosome
    maf_df["CHR-START"] = maf_df["CHR"]+"-"+(np.round(maf_df["START"]/maf_df["CHR_LENGTH"], rounding_factor)).astype(str)
    maf_df["CHR-END"] = maf_df["CHR"]+"-"+(np.round(maf_df["END"]/maf_df["CHR_LENGTH"], rounding_factor)).astype(str)
    train["CHR-START"] = train["CHR"]+"-"+(np.round(train["START"]/train["CHR_LENGTH"], rounding_factor)).astype(str)
    train["CHR-END"] = train["CHR"]+"-"+(np.round(train["END"]/train["CHR_LENGTH"], rounding_factor)).astype(str)
    
    # Pour la feature EFFECT, il y a peu de valeurs différentes, on peut donc faire un one-hot encoding
    effect_dummies = pd.get_dummies(maf_df["EFFECT"])
    maf_df = pd.concat([maf_df, effect_dummies], axis=1)
    effect_counts = maf_df.groupby("ID").aggregate({change: "sum" for change in effect_dummies.columns})
    
    features_to_encode = ["PROTEIN_CHANGE","REFALT", "CHR-START"]
    for feature in features_to_encode:
        grouped = train.groupby(feature)["OS_YEARS"].aggregate(lambda x: {"mean": x.mean(), "weight": x.count()/((x.std() if not np.isnan(x.var()) else 0)+1)})
        encoded = pd.json_normalize(grouped).set_index(grouped.index)
        encoded["weighted_mean"] = encoded["mean"] * encoded["weight"]
        encoded = encoded.add_prefix(f"{feature}_")
        maf_df = maf_df.merge(encoded, left_on=feature, right_index=True, how="left")
        maf_df.drop(feature, axis=1, inplace=True)
      
    count = maf_df.groupby("ID").size().reset_index(name='Nmut')  

    maf_df = maf_df.groupby("ID").aggregate({
        **{"VAF": "mean", "DEPTH": "mean"}, 
        **{feature+"_weighted_mean": "sum" for feature in features_to_encode},
        **{feature+"_mean": lambda x: {"sum": x.sum(), "min": x.min(), "max": x.max(), "std": x.std(), "skew": x.skew(), "median": x.median()} for feature in features_to_encode},
        **{feature+"_weight": "sum" for feature in features_to_encode},
    })
        
    maf_df = maf_df.merge(count, on="ID")
    maf_df = maf_df.merge(effect_counts, on="ID")

    maf_df = flatten_dict_columns(maf_df)

    
    for feature in features_to_encode:
        maf_df[feature] = maf_df[feature+"_weighted_mean"] / maf_df[feature+"_weight"]
        maf_df.drop([feature+"_weighted_mean", feature+"_weight"], axis=1, inplace=True)
    return maf_df