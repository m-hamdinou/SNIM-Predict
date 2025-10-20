import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import plotly.express as px
from datetime import datetime
from io import BytesIO
import base64
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import joblib
import os
import csv

# ==========================================================
# CONFIGURATION
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
st.write("_IA de maintenance prédictive développée pour la SNIM par **HAMDINOU Moulaye Driss**_")

# ==========================================================
# CHARGEMENT DES DONNÉES INTERNES
# ==========================================================
data_dir = "data"
train_path = os.path.join(data_dir, "aps_failure_training_set.csv")
test_path = os.path.join(data_dir, "aps_failure_test_set.csv")

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    st.error("⚠️ Données internes Scania APS introuvables. Vérifie le dossier /data.")
    st.stop()

# Détection automatique du séparateur (, ou ;)
def auto_read_csv(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(1024)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        return pd.read_csv(path, sep=dialect.delimiter, engine="python", encoding="utf-8")

st.info("📂 Chargement et nettoyage des données internes...")
train_df = auto_read_csv(train_path)
test_df = auto_read_csv(test_path)

for df in [train_df, test_df]:
    df.replace("na", np.nan, inplace=True)
    num_cols = df.columns.difference(["class"])
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["class"] = df["class"].map({"neg": 0, "pos": 1})

X_train, y_train = train_df.drop(columns=["class"]), train_df["class"]
X_test, y_test = test_df.drop(columns=["class"]), test_df["class"]

# ==========================================================
# ENTRAÎNEMENT
# ==========================================================
st.info("🚀 Entraînement du modèle Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)
joblib.dump(model, "snim_model.pkl")

# ==========================================================
# ÉVALUATION
# ==========================================================
st.info("🔮 Évaluation du modèle sur le jeu de test...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ==========================================================
# INTERFACE
# ==========================================================
mode = st.sidebar.radio("🧭 Mode :", ["Vue Synthétique", "Mode Technique"])
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 SNIM Predict – Développée par **HAMDINOU Moulaye Driss**")

if mode == "Vue Synthétique":
    st.subheader("📊 Résultats du diagnostic global")
    st.metric("Exactitude (Accuracy)", f"{acc:.3f}")
    st.metric("Score F1", f"{f1:.3f}")

    st.markdown("### 📉 Matrice de confusion")
    cm_df = pd.DataFrame(cm, columns=["Prédit négatif", "Prédit positif"], index=["Réel négatif", "Réel positif"])
    st.dataframe(cm_df)

    st.markdown("### 🧭 Schéma de fonctionnement")
    mermaid = """
    graph TD
    A[Capteurs IoT sur engins] --> B[Prétraitement]
    B --> C[Modèle IA Random Forest]
    C --> D{Diagnostic prédictif}
    D -->|🟢 Normal| E[OK]
    D -->|🟠 Dérive| F[Surveillance]
    D -->|🔴 Panne| G[Intervention]
    """
    st.markdown(f"```mermaid\n{mermaid}\n```")

    if st.button("📄 Générer le rapport PDF"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        if os.path.exists("snim_logo.png"):
            story.append(Image("snim_logo.png", width=120, height=60))
        story.append(Spacer(1, 15))
        story.append(Paragraph("<b>Rapport SNIM Predict</b>", styles["Title"]))
        story.append(Spacer(1, 15))
        story.append(Paragraph(f"Accuracy : {acc:.3f} | F1-score : {f1:.3f}", styles["Normal"]))
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

else:
    st.subheader("🔬 Détails techniques du modèle")
    st.text(classification_report(y_test, y_pred))
    imp = pd.DataFrame({"Variable": X_train.columns, "Importance": model.feature_importances_})
    fig_imp = px.bar(imp.nlargest(20, "Importance"), x="Variable", y="Importance", title="Top 20 variables importantes")
    st.plotly_chart(fig_imp, use_container_width=True)
