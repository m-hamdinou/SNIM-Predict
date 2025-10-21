# ==========================================================
# SNIM Predict – IA de maintenance prédictive
# Développée par HAMDINOU Moulaye Driss (© 2025)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# ==========================================================
# CONFIGURATION
# ==========================================================
st.set_page_config(page_title="SNIM Predict", page_icon="🤖", layout="wide")

st.markdown("""
<style>
h1, h2, h3 {color:#00b4d8;}
.main {background-color:#0d1117; color:#e0e0e0;}
</style>
""", unsafe_allow_html=True)

if "snim_logo.png" in [f for f in st.session_state.get("uploaded_files", [])] or True:
    st.image("snim_logo.png", width=160)

st.title("💡 SNIM Predict – Supervision & Diagnostic Intelligent")
st.write("_IA de maintenance prédictive développée pour la SNIM par **HAMDINOU Moulaye Driss**_")

# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================
drive_url = "https://drive.google.com/uc?id=14RZB_Qe62IJnB_b86o0fhlzmduLjLUGi"
st.info("📥 Chargement des données depuis Google Drive...")
response = requests.get(drive_url)
df = pd.read_csv(StringIO(response.text))

st.success(f"✅ Données chargées : {len(df):,} lignes – {df['device'].nunique()} engins détectés.")
st.write(df.head())

# ==========================================================
# PRÉTRAITEMENT
# ==========================================================
df = df.dropna()
metric_cols = [c for c in df.columns if "metric" in c]

X = df[metric_cols]
y = df["failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================================
# MODÈLE XGBOOST
# ==========================================================
model = XGBClassifier(
    tree_method="hist",  # compatible CPU/GPU
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=20,  # très important pour classes déséquilibrées
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# ==========================================================
# 📊 PERFORMANCE INTERACTIVE
# ==========================================================
st.header("📊 Performances globales du modèle")

threshold = st.slider("🎚️ Seuil de détection de panne (sensibilité)",
                      0.05, 0.9, 0.5, 0.05)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred_adj = (y_proba >= threshold).astype(int)

acc_adj = accuracy_score(y_test, y_pred_adj)
f1_adj = f1_score(y_test, y_pred_adj)
cm = confusion_matrix(y_test, y_pred_adj)

col1, col2 = st.columns(2)
col1.metric("Exactitude (Accuracy)", f"{acc_adj:.3f}")
col2.metric("Score F1", f"{f1_adj:.3f}")

st.write("**Matrice de confusion :**")
st.dataframe(pd.DataFrame(cm, index=["Normal", "Panne"], columns=["Prévu normal", "Prévu panne"]))

# ==========================================================
# 🧠 ANALYSE DES PANNES DÉTECTÉES
# ==========================================================
st.header("🧠 Analyse explicative des pannes détectées")

df["predicted_failure"] = model.predict_proba(scaler.transform(df[metric_cols]))[:, 1]
df["prediction_label"] = (df["predicted_failure"] >= threshold).astype(int)

engins_risque = df[df["prediction_label"] == 1]
nb_risque = engins_risque["device"].nunique()
st.write(f"🚨 {nb_risque} engins actuellement considérés en risque (seuil={threshold}).")

if nb_risque > 0:
    mean_failure = engins_risque[metric_cols].mean()
    mean_normal = df[df["prediction_label"] == 0][metric_cols].mean()
    diff = (mean_failure - mean_normal).sort_values(ascending=False)

    diff_df = pd.DataFrame(diff, columns=["Différence Moyenne"])
    diff_df["Importance relative (%)"] = 100 * diff_df["Différence Moyenne"].abs() / diff_df["Différence Moyenne"].abs().sum()

    st.markdown("#### 🔍 Capteurs les plus influents dans les engins à risque")
    st.dataframe(diff_df.head(10).style.background_gradient(cmap="Reds"))

    fig_diff = px.bar(diff_df.head(10).reset_index(),
                      x="index", y="Importance relative (%)",
                      title="Variables les plus associées aux pannes détectées",
                      color="Importance relative (%)")
    st.plotly_chart(fig_diff, use_container_width=True)
else:
    st.info("✅ Aucun engin à risque détecté avec ce seuil.")

# ==========================================================
# ⚙️ MODE EXPERT – SURVEILLANCE TEMPS RÉEL
# ==========================================================
st.header("⚙️ Mode Expert – Surveillance en temps réel")

recent_failures = df[df["prediction_label"] == 1].sort_values(by="predicted_failure", ascending=False).head(10)
if len(recent_failures) > 0:
    st.write("### 🔴 Derniers engins à risque détectés")
    st.dataframe(recent_failures[["date", "device", "predicted_failure"] + metric_cols[:3]])
else:
    st.success("🟢 Aucun signal d’anomalie critique détecté actuellement.")

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:gray;font-size:12px;">© 2025 SNIM Predict – Développée par HAMDINOU Moulaye Driss</div>',
    unsafe_allow_html=True
)
