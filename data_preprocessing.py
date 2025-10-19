import pandas as pd
import numpy as np
import streamlit as st

# Colonnes attendues dans les fichiers IoT
COLONNES_ATTENDUES = ["Engin", "Temperature", "Vibration", "Pression", "Label"]

def valider_et_preparer(df):
    """
    Vérifie la validité et prépare le DataFrame IoT pour l'analyse.
    Retourne un tuple : (dataframe_nettoye, message)
    """

    # === 1️⃣ Vérification des colonnes ===
    colonnes_manquantes = [col for col in COLONNES_ATTENDUES if col not in df.columns]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes : {', '.join(colonnes_manquantes)}")

    # === 2️⃣ Nettoyage des doublons et valeurs manquantes ===
    nb_initial = len(df)
    df = df.drop_duplicates().dropna()
    nb_supprime = nb_initial - len(df)

    # === 3️⃣ Conversion automatique des types ===
    for col in ["Temperature", "Vibration", "Pression", "Label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()  # retire les lignes non convertibles

    # === 4️⃣ Vérification des valeurs hors normes ===
    conditions = (
        (df["Temperature"] >= -20) & (df["Temperature"] <= 150) &
        (df["Vibration"] >= 0) & (df["Vibration"] <= 5) &
        (df["Pression"] >= 0) & (df["Pression"] <= 10)
    )
    lignes_valides = df[conditions]
    lignes_invalides = len(df) - len(lignes_valides)

    # === 5️⃣ Normalisation facultative ===
    # On peut normaliser les colonnes pour que le modèle soit plus stable
    df_norm = lignes_valides.copy()
    for col in ["Temperature", "Vibration", "Pression"]:
        df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()

    # === 6️⃣ Message de validation ===
    msg = f"""
    ✅ Données prêtes pour l'analyse :
    - {len(df_norm)} lignes valides
    - {nb_supprime} doublons ou valeurs manquantes supprimées
    - {lignes_invalides} lignes hors normes retirées
    """

    return df_norm, msg


# Exemple d'utilisation directe (utile pour tester)
if __name__ == "__main__":
    fichier = "data_iot/capteurs_20251019_2230.csv"
    try:
        df = pd.read_csv(fichier)
        df_prep, message = valider_et_preparer(df)
        print(message)
        print(df_prep.head())
    except Exception as e:
        print(f"Erreur : {e}")
