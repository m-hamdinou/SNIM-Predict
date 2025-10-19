import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# === CONFIGURATION ===
DOSSIER_SORTIE = "data_iot"
NB_ENGINS = 10         # nombre de machines simulées
DELAI_SECONDES = 120   # délai entre chaque génération (2 min)

def generer_donnees_iot():
    """Génère un fichier CSV avec des données simulées de capteurs"""
    if not os.path.exists(DOSSIER_SORTIE):
        os.makedirs(DOSSIER_SORTIE)

    engins = np.arange(NB_ENGINS)
    temperature = np.random.normal(70, 5, NB_ENGINS)   # moyenne 70°C
    vibration = np.random.normal(0.3, 0.15, NB_ENGINS) # amplitude moyenne 0.3
    pression = np.random.normal(2.5, 0.4, NB_ENGINS)   # en bars

    # Label automatique : 1 si vibration élevée ou température trop haute
    label = ((temperature > 75) | (vibration > 0.6)).astype(int)

    df = pd.DataFrame({
        "Engin": engins,
        "Temperature": temperature.round(2),
        "Vibration": vibration.round(2),
        "Pression": pression.round(2),
        "Label": label
    })

    nom_fichier = f"capteurs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    chemin = os.path.join(DOSSIER_SORTIE, nom_fichier)
    df.to_csv(chemin, index=False)
    print(f"✅ Fichier généré : {chemin}")

def boucle_generation():
    """Boucle infinie pour générer les fichiers en continu"""
    print("=== Génération de données IoT en continu ===")
    print(f"Dossier : {DOSSIER_SORTIE}")
    print(f"Fréquence : 1 fichier toutes les {DELAI_SECONDES} secondes\n")

    while True:
        generer_donnees_iot()
        time.sleep(DELAI_SECONDES)

if __name__ == "__main__":
    boucle_generation()
