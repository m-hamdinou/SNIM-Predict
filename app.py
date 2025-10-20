import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
from datetime import datetime
from io import BytesIO
import base64
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os

# ==========================================================
# CONFIGURATION GÉNÉRALE
# ==========================================================
st.set_page_config(page_title="SNIM Predict", page_icon="🤖", layout="wide")

st.markdown("""
<style>
h1, h2, h3 {color:#004b8d;}
.main {background-color:#ffffff; padding:25px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

if os.path.exists("snim_logo.png"):
    st.image("snim_logo.png", width=160)

st.title("💡 SNIM Predict – Supervision & Diagnostic Intelligent")
st.write("_IA de maintenance prédictive développée par **HAMDINOU Moulaye Driss**_")

# ==========================================================
# INTRODUCTION
# ==========================================================
st.markdown("""
### 📖 Introduction

Les données utilisées proviennent d’un **jeu de données public de maintenance prédictive**,
simulant une flotte de **5 engins industriels** envoyant leurs relevés de capteurs.
L’objectif est de prédire la **probabilité de panne** (`failure = 1`) à partir des mesures.

Nous utilisons ici un **échantillon de 10 000 lignes**, représentatif mais léger,
permettant d’entraîner rapidement un modèle d’intelligence artificielle
pour le diagnostic préventif et la supervision en temps réel.
""")

# ==========================================================
# CHARGEMENT DU DATASET INTERNE
# ==========================================================
st.info("📂 Chargement du dataset interne (5 engins / 10 000 lignes)...")

try:
    df = pd.read_csv("subset_5_engins_10000.csv")
    st.success(f"✅ Données chargées : {df.shape[0]} lignes – {df['device'].nunique()} engins détectés")

    # Vérif colonnes
    expected_metrics = [c for c in df.columns if "metric" in c]
    if len(expected_metrics) == 0:
        st.error("❌ Aucune colonne 'metric' détectée.")
    else:
        # ==========================================================
        # ENTRAÎNEMENT DU MODÈLE
        # ==========================================================
        X = df[expected_metrics]
        y = df["failure"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.subheader("📊 Performances du modèle")
        col1, col2 = st.columns(2)
        col1.metric("Exactitude (Accuracy)", f"{acc:.3f}")
        col2.metric("Score F1", f"{f1:.3f}")

        # ==========================================================
        # ANALYSE PAR ENGIN
        # ==========================================================
        st.markdown("### 🏗️ Analyse par engin (device)")
        df["predicted_failure"] = model.predict(X)
        resume = df.groupby("device")["predicted_failure"].mean().reset_index()
        resume["Statut"] = resume["predicted_failure"].apply(
            lambda x: "🔴 Risque élevé" if x > 0.6 else ("🟠 Risque moyen" if x > 0.3 else "🟢 Normal")
        )

        fig = px.bar(
            resume,
            x="device",
            y="predicted_failure",
            color="Statut",
            color_discrete_map={"🔴 Risque élevé": "red", "🟠 Risque moyen": "orange", "🟢 Normal": "green"},
            title="Indice de risque moyen par engin"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================================
        # SCHÉMA EXPLICATIF (IA)
        # ==========================================================
        st.markdown("### 🧭 Schéma de fonctionnement de SNIM Predict")
        mermaid = """
        graph TD
        A[Capteurs IoT sur engins] --> B[Collecte & Prétraitement des signaux]
        B --> C[Modèle IA Random Forest]
        C --> D[Analyse des comportements]
        D --> E{Diagnostic prédictif}
        E -->|🟢 Normal| F[OK]
        E -->|🟠 Dérive| G[Surveillance]
        E -->|🔴 Panne| H[Intervention urgente]
        """
        st.markdown(f"```mermaid\n{mermaid}\n```")

        # ==========================================================
        # GÉNÉRATION DU RAPPORT PDF
        # ==========================================================
        if st.button("📄 Générer le rapport PDF"):
            try:
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []

                if os.path.exists("snim_logo.png"):
                    story.append(Image("snim_logo.png", width=120, height=60))
                story.append(Spacer(1, 15))
                story.append(Paragraph("<b>Rapport SNIM Predict</b>", styles["Title"]))
                story.append(Spacer(1, 15))
                story.append(Paragraph(f"Précision : {acc:.3f} | Score F1 : {f1:.3f}", styles["Normal"]))
                story.append(Spacer(1, 10))
                story.append(Paragraph("Résumé par engin :", styles["Heading3"]))

                table_data = [["Device", "Indice", "Statut"]] + resume.values.tolist()
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                story.append(table)
                story.append(Spacer(1, 20))
                story.append(Paragraph(
                    f"Analyse effectuée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}.<br/>"
                    "<b>IA développée par HAMDINOU Moulaye Driss – Data Scientist</b>",
                    styles["Italic"]
                ))

                doc.build(story)
                buffer.seek(0)
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="rapport_snim.pdf">📥 Télécharger le rapport PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Erreur lors de la génération du PDF : {e}")

except FileNotFoundError:
    st.error("❌ Le fichier 'subset_5_engins_10000.csv' est introuvable dans le dossier.")
