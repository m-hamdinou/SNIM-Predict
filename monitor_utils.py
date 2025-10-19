import pandas as pd
import os
import time

def lire_donnees_en_continu(dossier="data_iot", delai=120):
    """
    Surveille le dossier spécifié et renvoie les données concaténées
    dès qu’un nouveau fichier est détecté.
    """
    fichiers_vus = set()
    while True:
        if not os.path.exists(dossier):
            os.makedirs(dossier)
        fichiers = [f for f in os.listdir(dossier) if f.endswith(".csv")]
        nouveaux = [f for f in fichiers if f not in fichiers_vus]

        if nouveaux:
            dfs = [pd.read_csv(os.path.join(dossier, f)).dropna() for f in fichiers]
            df = pd.concat(dfs, ignore_index=True)
            fichiers_vus.update(fichiers)
            yield df  # renvoie les données actualisées

        time.sleep(delai)
