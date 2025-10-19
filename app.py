import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
from datetime import datetime
from rapport_utils import generer_pdf
from monitor_utils import lire_donnees_en_continu
import os
import threading

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="SNIM Predict – Supervision", page_icon="⚙️", layout="wide")

if os.path.exists("snim_logo.png"):
    st.image("snim_logo.png", width=150)
st.title("⚙️ SNIM Predict – Supervision en continu des engins")

st.info("Les données IoT sont lues en continu depuis le dossier `data_iot/` toutes les 2 minutes.")

placeholder = st.empty()

# ==================== FONCTION D'APPRENTISSAGE ====================
def entrainer_et_afficher(df):
    """Entraîne le modèle IA et met à jour les résultats à l’écran"""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.factorize(df[col])[0]

    X = df.drop(columns=["Label"]) if "Label" in df.columns else df.iloc[:, :-1]
    y = df["Label"] if "Label" in df.columns else df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    resume = pd.DataFrame()
    if "Engin" in df.columns:
        resume = df.groupby("Engin")["Label"].mean().reset_index()
        resume["Statut"] = resume["Label"].apply(
            lambda x: "⚠️ Risque élevé" if x > 0.6 else ("🔸 Risque moyen" if x > 0.3 else "✅ Normal")
        )

    with placeholder.container():
        st.markdown(f"### 📈 Précision : **{acc:.2f}** | Score F1 : **{f1:.2f}**")
        if not resume.empty:
            fig = px.bar(
                resume, x="Engin", y="Label", color="Statut",
                color_discrete_map={"⚠️ Risque élevé":"red","🔸 Risque moyen":"orange","✅ Normal":"green"},
                title="Niveau de risque par engin"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucun champ 'Engin' trouvé dans les données.")

        # Rapport PDF
        generer_pdf(acc, f1, resume)

# ==================== SUPERVISION EN CONTINU ====================
def boucle_supervision():
    for df in lire_donnees_en_continu("data_iot", delai=120):
        entrainer_et_afficher(df)

# Lancer la surveillance dans un thread séparé
threading.Thread(target=boucle_supervision, daemon=True).start()

st.markdown(
    '<div style="text-align:center;color:gray;font-size:12px;margin-top:30px;">'
    '© 2025 SNIM Predict – Développée par HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
