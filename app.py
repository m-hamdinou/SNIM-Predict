import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os

# ==================== CONFIGURATION DE LA PAGE ====================
st.set_page_config(page_title="SNIM Predict", page_icon="🤖", layout="centered")

st.markdown("""
<style>
body {background-color: #f8fafc;}
.main {background-color: #ffffff; border-radius: 12px; padding: 25px;
       box-shadow: 0 2px 8px rgba(0,0,0,0.08);}
h1, h2, h3 {color: #004b8d;}
.footer {text-align: center; color: gray; font-size: 13px; margin-top: 40px;}
</style>
""", unsafe_allow_html=True)

# ==================== ENTÊTE ====================
if os.path.exists("snim_logo.png"):
    st.image("snim_logo.png", width=160)
st.title("💡 SNIM Predict")
st.write("### Maintenance prédictive assistée par Intelligence Artificielle")
st.write("_Développée par HAMDINOU Moulaye Driss – Data Scientist_")

st.info(
    "SNIM Predict analyse les données IoT des machines pour détecter immédiatement "
    "les comportements anormaux. Les techniciens peuvent l’utiliser chaque jour "
    "ou chaque semaine pour anticiper les pannes et planifier les interventions."
)

# ==================== IMPORTATION DES DONNÉES ====================
uploaded_file = st.file_uploader("📂 Importez vos données IoT (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Aperçu des données :")
    st.dataframe(df.head())

    if st.button("🚀 Entraîner le modèle IA"):
        df = df.dropna().copy()

        # Encodage automatique des variables catégorielles
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.factorize(df[col])[0]

        # Séparation des variables et du label
        X = df.drop(columns=["Label"]) if "Label" in df.columns else df.iloc[:, :-1]
        y = df["Label"] if "Label" in df.columns else df.iloc[:, -1]
        X = X.select_dtypes(include=[np.number])
        y = y.astype(int)

        # Entraînement du modèle
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.success("✅ Modèle entraîné avec succès !")
        st.markdown(f"### 📈 Précision : **{acc:.2f}** | Score F1 : **{f1:.2f}**")

        # ==================== INTERPRÉTATION ====================
        st.markdown("#### 🧠 Interprétation automatique :")
        if acc >= 0.95:
            st.success("Le modèle détecte parfaitement les anomalies : fiabilité industrielle.")
        elif acc >= 0.80:
            st.warning("Bon modèle, mais un recalibrage pourrait encore améliorer la précision.")
        else:
            st.error("Précision faible : revoir la qualité ou la quantité de données capteurs.")

        # ==================== RÉSUMÉ PAR ENGIN ====================
        if "Engin" in df.columns:
            st.markdown("### 📊 État des engins :")
            resume = df.groupby("Engin")["Label"].mean().reset_index()
            resume["Statut"] = resume["Label"].apply(
                lambda x: "⚠️ Risque élevé" if x > 0.6 else ("🔸 Risque moyen" if x > 0.3 else "✅ Normal")
            )
            st.dataframe(resume[["Engin", "Statut", "Label"]])
        else:
            resume = pd.DataFrame()

        # ==================== GÉNÉRATION DU PDF ====================
       if st.button("📄 Générer le rapport PDF"):
         try:
               # --- création du PDF ---
               doc = SimpleDocTemplate("rapport_snim.pdf", pagesize=A4)
               styles = getSampleStyleSheet()
               story = []

          if os.path.exists("snim_logo.png"):
            story.append(Image("snim_logo.png", width=120, height=60))
               story.append(Spacer(1, 20))
               story.append(Paragraph("<b>Rapport SNIM Predict</b>", styles["Title"]))
               story.append(Spacer(1, 15))
               story.append(Paragraph(f"Précision : {acc:.2f} Score F1 : {f1:.2f}", styles["BodyText"]))
               story.append(Spacer(1, 10))

          if not resume.empty:
             story.append(Paragraph("<b>Résumé par engin :</b>", styles["Heading3"]))
             for _, row in resume.iterrows():
                story.append(Paragraph(f"{row['Engin']} – {row['Statut']}", styles["BodyText"]))
             story.append(Spacer(1, 15))

          story.append(Paragraph(
            f"Analyse effectuée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}<br/>"
            "<b>IA développée par HAMDINOU Moulaye Driss – Data Scientist</b>",
            styles["Italic"]
           ))
           doc.build(story)

        # --- lecture et envoi du PDF ---
        with open("rapport_snim.pdf", "rb") as f:
            pdf_data = f.read()

        st.download_button(
            "⬇️ Télécharger le rapport PDF",
            data=pdf_data,
            file_name="rapport_snim.pdf",
            mime="application/pdf",
            key="download_pdf"
        )

        st.success("✅ Rapport généré avec succès ! Cliquez sur le bouton pour le télécharger.")

  except Exception as e:
         st.error(f"⚠️ Erreur lors de la génération du rapport : {e}")

st.markdown(
    '<div class="footer">© 2025 SNIM Predict – HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
