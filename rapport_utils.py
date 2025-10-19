import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import tempfile, os

def generer_pdf(acc, f1, resume):
    """
    Génère un rapport PDF complet et retourne un bouton de téléchargement fonctionnel.
    Compatible Streamlit Cloud / Colab.
    """
    try:
        # === Fichier temporaire sûr ===
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = tmp_file.name

        # === Préparation du document ===
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # === Logo SNIM ===
        if os.path.exists("snim_logo.png"):
            story.append(Image("snim_logo.png", width=120, height=60))
        story.append(Spacer(1, 20))

        # === Titre ===
        story.append(Paragraph("<b><font size=16 color='#004b8d'>Rapport SNIM Predict</font></b>", styles["Title"]))
        story.append(Spacer(1, 12))

        # === Résumé global ===
        story.append(Paragraph(
            f"<b>Exactitude (Accuracy)</b> : {acc:.2f}<br/><b>Score F1</b> : {f1:.2f}",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 15))

        # === Tableau des engins ===
        if not resume.empty:
            story.append(Paragraph("<b>Résumé par engin :</b>", styles["Heading3"]))
            story.append(Spacer(1, 6))

            data = [["Engin", "Indice moyen", "Diagnostic"]]
            for _, r in resume.iterrows():
                data.append([
                    str(r["Engin"]),
                    f"{r['Label']:.2f}",
                    r.get("Prochain_risque", "N/A")
                ])

            table = Table(data, colWidths=[60, 80, 300])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#004b8d")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        else:
            story.append(Paragraph("Aucune donnée disponible.", styles["Normal"]))

        # === Pied de page ===
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f"Analyse effectuée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}<br/>"
            "<b>IA développée par HAMDINOU Moulaye Driss – Data Scientist</b>",
            styles["Italic"]
        ))

        # === Construction du PDF ===
        doc.build(story)

        # === Lecture et téléchargement ===
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        st.download_button(
            label="⬇️ Télécharger le rapport PDF",
            data=pdf_data,
            file_name="rapport_snim.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.success("✅ Rapport PDF généré avec succès !")

    except Exception as e:
        st.error(f"⚠️ Erreur lors de la génération du PDF : {e}")
