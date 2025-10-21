# ==========================================================
# 💡 SNIM Predict – IA industrielle avancée
# Développée par HAMDINOU Moulaye Driss
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os

# ==========================================================
# 🎨 CONFIGURATION INTERFACE
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
# 📥 CHARGEMENT DU DATASET (depuis Google Drive)
# ==========================================================
DRIVE_URL = "https://drive.google.com/uc?id=14RZB_Qe62IJnB_b86o0fhlzmduLjLUGi"

@st.cache_data(show_spinner=True)
def load_data():
    st.info("📥 Chargement des données depuis Google Drive…")
    df = pd.read_csv(DRIVE_URL)
    st.success(f"✅ Données chargées : {df.shape[0]} lignes – {df['device'].nunique()} engins détectés.")
    return df

df = load_data()

if not df.empty:
    # ==========================================================
    # 🧹 PRÉTRAITEMENT
    # ==========================================================
    metric_cols = [c for c in df.columns if "metric" in c]
    X = df[metric_cols]
    y = df["failure"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    # Rééquilibrage des classes avec SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # ==========================================================
    # 🧠 ENTRAÎNEMENT AVEC XGBoost
    # ==========================================================
    model = XGBClassifier(
        scale_pos_weight=(len(y_res) - sum(y_res)) / sum(y_res),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_res, y_res)

    # Prédiction
    y_pred = model.predict(X_test)
    acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)

    # ==========================================================
    # 📊 PERFORMANCES GLOBALES
    # ==========================================================
    st.subheader("📊 Performances globales du modèle")
    col1, col2 = st.columns(2)
    col1.metric("Exactitude (Accuracy)", f"{acc:.3f}")
    col2.metric("Score F1", f"{f1:.3f}")

    st.write("**Matrice de confusion :**")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=["Normal", "Panne"], columns=["Prévu normal", "Prévu panne"]))

    # ==========================================================
    # 🔍 ANALYSE DE CORRÉLATION
    # ==========================================================
    st.markdown("### 🔬 Corrélation entre capteurs et pannes")
    corr = df[metric_cols + ["failure"]].corr()["failure"].sort_values(ascending=False)
    fig_corr = px.bar(corr, title="Corrélation de chaque capteur avec la variable de panne")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ==========================================================
    # 🏗️ ANALYSE PAR ENGIN
    # ==========================================================
    st.markdown("### 🏗️ Analyse des engins et prédictions IA")
    df["predicted_failure"] = model.predict_proba(scaler.transform(df[metric_cols]))[:, 1]

    resume = df.groupby("device")["predicted_failure"].mean().reset_index()
    resume["Statut"] = resume["predicted_failure"].apply(
        lambda x: "🔴 Risque élevé" if x > 0.6 else ("🟠 Risque moyen" if x > 0.3 else "🟢 Normal")
    )

    fig = px.bar(
        resume, x="device", y="predicted_failure", color="Statut",
        color_discrete_map={"🔴 Risque élevé": "red", "🟠 Risque moyen": "orange", "🟢 Normal": "green"},
        title="Indice de risque moyen par engin"
    )
    st.plotly_chart(fig, use_container_width=True)

    problemes = resume[resume["Statut"] != "🟢 Normal"].sort_values("predicted_failure", ascending=False)
    st.markdown("### 🚨 Engins présentant un risque de panne")
    st.dataframe(problemes.head(20))

    # ==========================================================
    # ⏱️ VISUALISATION TEMPORELLE PAR ENGIN
    # ==========================================================
    st.markdown("### 📈 Visualisation temporelle d’un engin")
    engins_disponibles = df["device"].unique().tolist()
    choix_engin = st.selectbox("Sélectionnez un engin :", engins_disponibles)
    df_engin = df[df["device"] == choix_engin]
    fig_time = px.line(df_engin, x="date", y=metric_cols, title=f"Évolution des capteurs – Engin {choix_engin}")
    st.plotly_chart(fig_time, use_container_width=True)

    # ==========================================================
    # 📄 RAPPORT PDF
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
            story.append(Paragraph(f"Exactitude : {acc:.3f} | Score F1 : {f1:.3f}", styles["Normal"]))
            story.append(Spacer(1, 10))
            story.append(Paragraph("Résumé des engins à risque :", styles["Heading3"]))

            table_data = [["Device", "Indice", "Statut"]] + problemes.values.tolist()
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
                f"Analyse effectuée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}<br/>"
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

else:
    st.error("❌ Impossible de charger les données.")
