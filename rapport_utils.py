from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import os
import streamlit as st

def generer_pdf(acc, f1, resume):
    try:
        doc = SimpleDocTemplate("rapport_snim.pdf", pagesize=A4)
        styles = getSampleStyleSheet()
        bleu = "#004b8d"

        title_style = ParagraphStyle('title_style', parent=styles['Title'], textColor=bleu, fontSize=20, spaceAfter=20)
        normal = ParagraphStyle('normal', parent=styles['BodyText'], fontSize=11, leading=15)
        section = ParagraphStyle('section', parent=styles['Heading2'], textColor=bleu, spaceBefore=15, spaceAfter=10)

        story = []
        if os.path.exists("snim_logo.png"):
            story.append(Image("snim_logo.png", width=120, height=60))
        story.append(Spacer(1,15))
        story.append(Paragraph("Rapport SNIM Predict", title_style))
        story.append(Paragraph(
            f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y à %H:%M')}<br/><b>Généré par :</b> IA SNIM Predict",
            normal
        ))

        story.append(Paragraph("1️⃣  Résumé des performances du modèle", section))
        story.append(Paragraph(f"Précision : {acc:.2f} Score F1 : {f1:.2f}", normal))

        story.append(Paragraph("2️⃣  Analyse par engin", section))
        if not resume.empty:
            for _, r in resume.iterrows():
                story.append(Paragraph(f"<b>Engin {r['Engin']}</b> – {r['Statut']}", normal))
        else:
            story.append(Paragraph("Aucune donnée d'engin disponible.", normal))

        story.append(PageBreak())
        story.append(Paragraph("3️⃣  Interprétation et recommandations", section))
        story.append(Paragraph(
            "Les engins avec un score de risque supérieur à 0.6 nécessitent une vérification prioritaire.<br/>"
            "Ce rapport peut être complété par les tendances capteurs et les historiques de maintenance.",
            normal
        ))

        story.append(Spacer(1,20))
        story.append(Paragraph(
            "<i>IA développée par HAMDINOU Moulaye Driss – Data Scientist</i>", normal))

        doc.build(story)

        with open("rapport_snim.pdf","rb") as f:
            st.download_button("⬇️ Télécharger le rapport PDF", f, file_name="rapport_snim.pdf", mime="application/pdf")
            st.success("✅ Rapport PDF généré avec succès !")

    except Exception as e:
        st.error(f"Erreur PDF : {e}")
