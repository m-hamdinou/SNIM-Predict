import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
from datetime import datetime
from rapport_utils import generer_pdf
from data_preprocessing import valider_et_preparer
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
    st.image("snim_logo.png", width=160)
st.title("💡 SNIM Predict – Supervision et Diagnostic Intelligent des Engins")
st.write("_Développée par **HAMDINOU Moulaye Driss – Data Scientist**_")

st.info(
    "Importez un ou plusieurs fichiers IoT pour analyser l’état des engins. "
    "L’IA SNIM Predict détecte automatiquement les comportements anormaux, "
    "prédit les pannes probables et génère un rapport clair."
)

# ==================== UPLOAD ====================
uploaded_files = st.file_uploader(
    "📂 Importez un ou plusieurs fichiers CSV de données capteurs",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ {len(uploaded_files)} fichier(s) importé(s) avec succès !")
    st.dataframe(df.head())

    # ==================== VALIDATION DES DONNÉES ====================
    try:
        df, message = valider_et_preparer(df)
        st.success(message)
    except Exception as e:
        st.error(f"❌ Erreur dans les données : {e}")
        st.stop()

    # ==================== ANALYSE IA ====================
    if st.button("🚀 Lancer l’analyse IA"):
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

        st.markdown(f"### 📊 Résultats de l’analyse")
        st.markdown(f"- **Exactitude (Accuracy)** : `{acc:.2f}`")
        st.markdown(f"- **Score F1** : `{f1:.2f}`")

        # ==================== RÉSUMÉ DES ENGINES ====================
        if "Engin" in df.columns:
            resume = df.groupby("Engin")["Label"].mean().reset_index()
            resume["Statut"] = resume["Label"].apply(
                lambda x: "⚠️ Risque élevé" if x > 0.6 else
                          ("🔸 Risque moyen" if x > 0.3 else "✅ Normal")
            )

            # Diagnostic automatique
            resume["Prochain_risque"] = resume["Label"].apply(
                lambda x: (
                    "🔴 À vérifier immédiatement (panne probable)" if x > 0.5 else
                    "🟠 Surveillance conseillée (début d’anomalie)" if x > 0.2 else
                    "🟢 OK – fonctionnement normal"
                )
            )

            st.markdown("### 🔍 Diagnostic automatique par engin")
            st.dataframe(resume[["Engin", "Label", "Prochain_risque"]])

            # Message d’alerte global
            engin_max = resume.loc[resume["Label"].idxmax()]
            if engin_max["Label"] > 0.5:
                st.error(f"🚨 L'engin {int(engin_max['Engin'])} présente un risque élevé ({engin_max['Label']:.2f}) — vérification urgente requise !")
            elif engin_max["Label"] > 0.2:
                st.warning(f"⚠️ L'engin {int(engin_max['Engin'])} montre une dérive possible ({engin_max['Label']:.2f}) — surveillance recommandée .")
            else:
                st.success("✅ Tous les engins fonctionnent normalement pour le moment.")

            # Graphique
            st.markdown("### 📈 Niveau de risque par engin")
            fig = px.bar(
                resume, x="Engin", y="Label", color="Prochain_risque",
                color_discrete_map={
                    "🔴 À vérifier immédiatement (panne probable)": "red",
                    "🟠 Surveillance conseillée (début d’anomalie)": "orange",
                    "🟢 OK – fonctionnement normal": "green"
                },
                title="Indice de risque global par engin"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            resume = pd.DataFrame()
            st.warning("⚠️ Aucune colonne 'Engin' détectée. Impossible de générer le diagnostic individuel.")

        # ==================== GÉNÉRATION PDF ====================
        if st.button("📄 Générer le rapport PDF"):
            try:
                generer_pdf(acc, f1, resume)
                st.success("✅ Rapport PDF généré ! Téléchargez-le ci-dessous :")
            except Exception as e:
                st.error(f"⚠️ Erreur lors de la génération du rapport : {e}")

st.markdown(
    '<div style="text-align:center;color:gray;font-size:12px;margin-top:40px;">'
    '© 2025 SNIM Predict – Développée par HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
