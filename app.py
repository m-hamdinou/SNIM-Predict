import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
from datetime import datetime
from rapport_utils import generer_pdf
import os

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="SNIM Predict", page_icon="🤖", layout="wide")

st.markdown("""
<style>
h1, h2, h3 {color:#004b8d;}
.main {background-color:#ffffff; padding:25px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

if os.path.exists("snim_logo.png"):
    st.image("snim_logo.png", width=150)
st.title("💡 SNIM Predict – Analyse de fichiers IoT")
st.info("Importez un ou plusieurs fichiers CSV contenant les données IoT des engins pour analyse immédiate.")

# ==================== IMPORTATION DES FICHIERS ====================
uploaded_files = st.file_uploader(
    "📂 Importez un ou plusieurs fichiers CSV", 
    type=["csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    dfs = [pd.read_csv(f).dropna() for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    st.success(f"{len(uploaded_files)} fichier(s) importé(s) avec succès ✅")
    st.dataframe(df.head())

    # ==================== ENTRAÎNEMENT IA ====================
    if st.button("🚀 Analyser les données IoT"):
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

        st.markdown(f"### 📈 Précision : **{acc:.2f}** | Score F1 : **{f1:.2f}**")

        # ==================== RÉSUMÉ DES ENGINES ====================
        if "Engin" in df.columns:
            resume = df.groupby("Engin")["Label"].mean().reset_index()
            resume["Statut"] = resume["Label"].apply(
                lambda x: "⚠️ Risque élevé" if x > 0.6 else ("🔸 Risque moyen" if x > 0.3 else "✅ Normal")
            )

            fig = px.bar(
                resume, x="Engin", y="Label", color="Statut",
                color_discrete_map={"⚠️ Risque élevé":"red","🔸 Risque moyen":"orange","✅ Normal":"green"},
                title="Niveau de risque par engin"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            resume = pd.DataFrame()
            st.warning("⚠️ Aucune colonne 'Engin' détectée dans le fichier.")

        # ==================== GÉNÉRATION PDF ====================
        if st.button("📄 Générer le rapport PDF"):
            generer_pdf(acc, f1, resume)

st.markdown(
    '<div style="text-align:center;color:gray;font-size:12px;margin-top:30px;">'
    '© 2025 SNIM Predict – Développée par HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
