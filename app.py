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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import joblib
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

# ==========================================================
# EN-TÊTE
# ==========================================================
if os.path.exists("snim_logo.png"):
    st.image("snim_logo.png", width=160)
st.title("💡 SNIM Predict – Supervision & Diagnostic Intelligent")
st.write("_IA de maintenance prédictive développée pour la SNIM par **HAMDINOU Moulaye Driss**_")

st.sidebar.title("🧭 Options")
mode = st.sidebar.radio("Choisir la vue :", ["Vue Synthétique", "Mode Technique"])
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 SNIM Predict – Développée par **HAMDINOU Moulaye Driss**")

# ==========================================================
# CHARGEMENT / UPLOAD DES DONNÉES
# ==========================================================
train_path = "aps_failure_training_set.csv"
test_path = "aps_failure_test_set.csv"

train_df = None
test_df = None

if os.path.exists(train_path) and os.path.exists(test_path):
    st.success("✅ Jeux de données Scania APS détectés automatiquement.")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
else:
    st.warning("📂 Fichiers Scania APS introuvables. Veuillez les importer manuellement ci-dessous :")
    uploaded_train = st.file_uploader("⬆️ Importer le fichier d'entraînement (aps_failure_training_set.csv)", type="csv")
    uploaded_test = st.file_uploader("⬆️ Importer le fichier de test (aps_failure_test_set.csv)", type="csv")

    if uploaded_train and uploaded_test:
        train_df = pd.read_csv(uploaded_train)
        test_df = pd.read_csv(uploaded_test)
        st.success("✅ Les deux fichiers ont été importés avec succès.")
    else:
        st.stop()

# ==========================================================
# PRÉTRAITEMENT
# ==========================================================
st.info("🧹 Nettoyage des données...")
for df in [train_df, test_df]:
    df.replace("na", np.nan, inplace=True)
    df[df.columns.difference(["class"])] = df[df.columns.difference(["class"])].astype("float32")
    df.fillna(df.median(), inplace=True)
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
# VUE SYNTHÉTIQUE
# ==========================================================
if mode == "Vue Synthétique":
    st.subheader("📊 Résultats du diagnostic global")
    st.metric("Exactitude (Accuracy)", f"{acc:.3f}")
    st.metric("Score F1", f"{f1:.3f}")

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

    st.markdown("### 📋 Matrice de confusion")
    cm_df = pd.DataFrame(cm, columns=["Prédit négatif", "Prédit positif"], index=["Réel négatif", "Réel positif"])
    st.dataframe(cm_df)

    # --- PDF ---
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
            story.append(Paragraph(f"Accuracy : {acc:.3f} | F1-score : {f1:.3f}", styles["Normal"]))
            story.append(Spacer(1, 10))
            story.append(Paragraph("Modèle : Random Forest (Scania APS Dataset)", styles["Normal"]))
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

# ==========================================================
# MODE TECHNIQUE
# ==========================================================
else:
    st.subheader("🔬 Analyse technique du modèle")
    st.write(f"**Accuracy :** {acc:.3f}   |   **F1 :** {f1:.3f}")
    st.write("**Rapport de classification :**")
    st.text(classification_report(y_test, y_pred))

    imp = pd.DataFrame({"Variable": X_train.columns, "Importance": model.feature_importances_})
    fig_imp = px.bar(imp.nlargest(20, "Importance"), x="Variable", y="Importance", title="Top 20 variables importantes")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 📉 Matrice de confusion détaillée")
    cm_df = pd.DataFrame(cm, columns=["Prédit négatif", "Prédit positif"], index=["Réel négatif", "Réel positif"])
    st.dataframe(cm_df)

# ==========================================================
# PIED DE PAGE
# ==========================================================
st.markdown(
    '<div style="text-align:center;color:gray;font-size:12px;margin-top:40px;">'
    '© 2025 SNIM Predict – Développée par HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
