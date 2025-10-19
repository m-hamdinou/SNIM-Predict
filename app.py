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
import tempfile
import os

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

# ==========================================================
# MENU LATÉRAL
# ==========================================================
mode = st.sidebar.selectbox("🧭 Choisir le mode :", ["Vue Synthétique", "Mode Technique"])
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 SNIM Predict – Développée par **HAMDINOU Moulaye Driss**")

# ==========================================================
# EN-TÊTE
# ==========================================================
if os.path.exists("snim_logo.png"):
    st.image("snim_logo.png", width=160)
st.title("💡 SNIM Predict – Supervision & Diagnostic Intelligent")
st.write("_IA de maintenance prédictive développée pour la SNIM par **HAMDINOU Moulaye Driss**_")

# ==========================================================
# UPLOAD DES FICHIERS
# ==========================================================
uploaded_files = st.file_uploader(
    "📂 Importez vos fichiers IoT (CSV)", 
    type=["csv"], 
    accept_multiple_files=True
)

# ==========================================================
# TRAITEMENT
# ==========================================================
if uploaded_files:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True).dropna()

    st.success(f"✅ {len(uploaded_files)} fichier(s) importé(s) avec succès.")
    st.dataframe(df.head())

    # Encodage automatique
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.factorize(df[c])[0]

    # Données
    X = df.drop(columns=["Label"]) if "Label" in df.columns else df.iloc[:, :-1]
    y = df["Label"] if "Label" in df.columns else df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)

    # ==========================================================
    # MODE SYNTHÉTIQUE
    # ==========================================================
    if mode == "Vue Synthétique":
        st.subheader("📊 Résumé général du diagnostic")
        st.metric("Exactitude (Accuracy)", f"{acc:.2f}")
        st.metric("Score F1", f"{f1:.2f}")

        if "Engin" in df.columns:
            resume = df.groupby("Engin")["Label"].mean().reset_index()
            resume["Statut"] = resume["Label"].apply(
                lambda x: "🔴 Risque élevé" if x > 0.6 else ("🟠 Risque moyen" if x > 0.3 else "🟢 Normal")
            )
            st.markdown("### 📈 État global des engins")
            fig = px.bar(
                resume, x="Engin", y="Label", color="Statut",
                color_discrete_map={"🔴 Risque élevé": "red", "🟠 Risque moyen": "orange", "🟢 Normal": "green"},
                title="Indice de risque par engin"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Diagnostic texte
            st.markdown("### 🔍 Diagnostic global")
            engin_max = resume.loc[resume["Label"].idxmax()]
            if engin_max["Label"] > 0.6:
                st.error(f"🚨 L'engin {int(engin_max['Engin'])} présente un risque de panne imminent.")
            elif engin_max["Label"] > 0.3:
                st.warning(f"⚠️ L'engin {int(engin_max['Engin'])} montre une dérive — à surveiller.")
            else:
                st.success("✅ Tous les engins fonctionnent normalement.")

            # Schéma IA
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

            # Génération du PDF
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
                    story.append(Paragraph(f"Précision : {acc:.2f} | Score F1 : {f1:.2f}", styles["Normal"]))
                    story.append(Spacer(1, 10))
                    story.append(Paragraph("Résumé par engin :", styles["Heading3"]))

                    table_data = [["Engin", "Indice", "Statut"]] + resume.values.tolist()
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

    # ==========================================================
    # MODE TECHNIQUE
    # ==========================================================
    elif mode == "Mode Technique":
        st.subheader("🔬 Détails techniques du modèle et des données")

        st.markdown(f"**Exactitude :** {acc:.2f}   |   **F1 :** {f1:.2f}")
        st.dataframe(df.head())

        st.markdown("### ⚙️ Importance des variables")
        imp = pd.DataFrame({"Variable": X.columns, "Importance": model.feature_importances_})
        fig_imp = px.bar(imp, x="Variable", y="Importance", title="Importance des variables")
        st.plotly_chart(fig_imp, use_container_width=True)

        if "Engin" in df.columns:
            resume = df.groupby("Engin")["Label"].mean().reset_index()
            fig_risk = px.bar(resume, x="Engin", y="Label", title="Risque individuel")
            st.plotly_chart(fig_risk, use_container_width=True)

# ==========================================================
# PIED DE PAGE
# ==========================================================
st.markdown(
    '<div style="text-align:center;color:gray;font-size:12px;margin-top:40px;">'
    '© 2025 SNIM Predict – Développée par HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
