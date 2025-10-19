import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
from datetime import datetime
from rapport_utils import generer_pdf   # 📄 fonctions PDF pro
import os

st.set_page_config(page_title="SNIM Predict", page_icon="🤖", layout="wide")

st.markdown("""
<style>
h1, h2, h3 {color:#004b8d;}
.main {background-color:#ffffff; padding:25px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

if os.path.exists("snim_logo.png"):
    st.image("snim_logo.png", width=150)
st.title("💡 SNIM Predict — Maintenance prédictive IoT")

uploaded_files = st.file_uploader(
    "📂 Importez un ou plusieurs fichiers CSV", type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    dfs = [pd.read_csv(f).dropna() for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    st.success(f"{len(uploaded_files)} fichiers importés et fusionnés ✅")
    st.dataframe(df.head())

    if st.button("🚀 Entraîner le modèle IA"):
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
        st.session_state.update({"acc": acc, "f1": f1, "df": df})

        st.success(f"Modèle entraîné ✅  |  Précision : {acc:.2f}  |  F1 : {f1:.2f}")

        if "Engin" in df.columns:
            resume = df.groupby("Engin")["Label"].mean().reset_index()
            resume["Statut"] = resume["Label"].apply(
                lambda x: "⚠️ Risque élevé" if x > 0.6 else ("🔸 Risque moyen" if x > 0.3 else "✅ Normal")
            )
            st.session_state["resume"] = resume

            # === Graphique Plotly ===
            fig = px.bar(
                resume, x="Engin", y="Label", color="Statut",
                color_discrete_map={"⚠️ Risque élevé":"red","🔸 Risque moyen":"orange","✅ Normal":"green"},
                title="Niveau de risque par engin", labels={"Label":"Score de risque"}
            )
            st.plotly_chart(fig, use_container_width=True)

        # === Prévision simple ===
        if "Temperature" in df.columns:
            df["Prévision"] = df["Temperature"].rolling(window=5).mean()
            st.line_chart(df[["Temperature","Prévision"]])

if "acc" in st.session_state and st.button("📄 Générer le rapport PDF"):
    generer_pdf(st.session_state["acc"], st.session_state["f1"], st.session_state.get("resume", pd.DataFrame()))
