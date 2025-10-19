import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
from datetime import datetime
import os, tempfile, base64
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ==================== CONFIG ====================
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

st.info("Importez un ou plusieurs fichiers CSV contenant les données IoT des engins pour une analyse immédiate et un rapport PDF complet.")

# ==================== UPLOAD ====================
uploaded_files = st.file_uploader("📂 Importez vos fichiers CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ {len(uploaded_files)} fichier(s) importé(s) avec succès !")
    st.dataframe(df.head())

    # Nettoyage
    df = df.dropna()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.factorize(df[col])[0]

    if st.button("🚀 Lancer l’analyse IA"):
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

        # === Résumé ===
        if "Engin" in df.columns:
            resume = df.groupby("Engin")["Label"].mean().reset_index()
            resume["Prochain_risque"] = resume["Label"].apply(
                lambda x: (
                    "🔴 À vérifier immédiatement" if x > 0.5 else
                    "🟠 Surveiller" if x > 0.2 else
                    "🟢 OK"
                )
            )
            st.markdown("### 🔍 Diagnostic automatique par engin")
            st.dataframe(resume)

            fig = px.bar(
                resume, x="Engin", y="Label", color="Prochain_risque",
                color_discrete_map={"🔴 À vérifier immédiatement":"red","🟠 Surveiller":"orange","🟢 OK":"green"},
                title="Indice de risque global par engin"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Aucune colonne 'Engin' détectée.")
            resume = pd.DataFrame()

        # ==================== GÉNÉRATION PDF ====================
        if st.button("📄 Générer le rapport PDF"):
            try:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf_path = tmp_file.name

                doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []

                if os.path.exists("snim_logo.png"):
                    story.append(Image("snim_logo.png", width=120, height=60))
                story.append(Spacer(1, 20))
                story.append(Paragraph("<b><font size=16 color='#004b8d'>Rapport SNIM Predict</font></b>", styles["Title"]))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"<b>Exactitude (Accuracy)</b> : {acc:.2f}<br/><b>Score F1</b> : {f1:.2f}", styles["BodyText"]))
                story.append(Spacer(1, 15))

                if not resume.empty:
                    story.append(Paragraph("<b>Résumé par engin :</b>", styles["Heading3"]))
                    data = [["Engin", "Indice moyen", "Diagnostic"]]
                    for _, r in resume.iterrows():
                        data.append([str(r["Engin"]), f"{r['Label']:.2f}", r["Prochain_risque"]])
                    table = Table(data, colWidths=[60, 80, 300])
                    table.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#004b8d")),
                        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                        ("ALIGN", (0,0), (-1,-1), "CENTER"),
                        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                        ("FONTSIZE", (0,0), (-1,-1), 10),
                        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
                    ]))
                    story.append(Spacer(1, 6))
                    story.append(table)

                story.append(Spacer(1, 20))
                story.append(Paragraph(
                    f"Analyse effectuée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}<br/>"
                    "<b>IA développée par HAMDINOU Moulaye Driss – Data Scientist</b>",
                    styles["Italic"]
                ))

                doc.build(story)

                # Conversion en base64 pour forcer le téléchargement
                with open(pdf_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

                href = f'<a href="data:application/octet-stream;base64,{b64}" download="rapport_snim.pdf">📥 Télécharger le rapport PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

                st.success("✅ Rapport PDF généré avec succès !")

            except Exception as e:
                st.error(f"⚠️ Erreur lors de la génération du rapport : {e}")

st.markdown(
    '<div style="text-align:center;color:gray;font-size:12px;margin-top:40px;">'
    '© 2025 SNIM Predict – Développée par HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
