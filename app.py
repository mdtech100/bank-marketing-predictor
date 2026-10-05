# app_tp2.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Configuration de la page ──────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Chargement des artefacts ──────────────────────────────────
@st.cache_resource
def charger_artefacts():
    modele    = joblib.load("gb_model.joblib")
    encoders  = joblib.load("encoders_tp2.joblib")
    scaler    = joblib.load("scaler.joblib")
    uniques   = joblib.load("uniques_tp2.joblib")
    return modele, encoders, scaler, uniques

modele, encoders, scaler, uniques = charger_artefacts()

# ── En-tête ───────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding:1.5rem 0 0.5rem 0'>
        <h1 style='color:#1a73e8; font-size:2.2rem'>🏦 Bank Marketing Predictor</h1>
        <p style='color:#555; font-size:1rem'>
            Prédisez si un client souscrira à un
            <b>dépôt à terme</b> suite à une campagne marketing
        </p>
    </div>
    <hr style='margin-bottom:1.5rem'>
""", unsafe_allow_html=True)

# ── Formulaire ────────────────────────────────────────────────
st.subheader("👤 Informations client")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🎂 Âge", min_value=18, max_value=100, value=35, step=1
    )
    job = st.selectbox("💼 Profession", options=uniques['job'])
    marital = st.selectbox("💍 Statut marital", options=uniques['marital'])
    education = st.selectbox("🎓 Niveau d'éducation", options=uniques['education'])
    housing = st.selectbox("🏠 Prêt immobilier", options=uniques['housing'])
    loan = st.selectbox("💳 Prêt personnel", options=uniques['loan'])
    contact = st.selectbox("📞 Type de contact", options=uniques['contact'])

with col2:
    month = st.selectbox("📅 Mois du dernier contact", options=uniques['month'])
    day_of_week = st.selectbox("📆 Jour de la semaine", options=uniques['day_of_week'])
    duration = st.number_input(
        "⏱️ Durée du dernier appel (sec)",
        min_value=0, max_value=5000, value=200, step=10
    )
    campaign = st.number_input(
        "📢 Nb contacts cette campagne",
        min_value=1, max_value=50, value=2, step=1
    )
    pdays = st.number_input(
        "📊 Jours depuis dernier contact (999 = jamais)",
        min_value=0, max_value=999, value=999, step=1
    )
    previous = st.number_input(
        "🔁 Nb contacts campagnes précédentes",
        min_value=0, max_value=50, value=0, step=1
    )
    poutcome = st.selectbox(
        "📋 Résultat campagne précédente",
        options=uniques['poutcome']
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Prédiction ────────────────────────────────────────────────
if st.button("🔍 Lancer la prédiction", use_container_width=True, type="primary"):

    # Encodage des variables catégorielles
    try:
        job_enc        = encoders['job'].transform([job])[0]
        marital_enc    = encoders['marital'].transform([marital])[0]
        education_enc  = encoders['education'].transform([education])[0]
        housing_enc    = encoders['housing'].transform([housing])[0]
        loan_enc       = encoders['loan'].transform([loan])[0]
        contact_enc    = encoders['contact'].transform([contact])[0]
        month_enc      = encoders['month'].transform([month])[0]
        day_enc        = encoders['day_of_week'].transform([day_of_week])[0]
        poutcome_enc   = encoders['poutcome'].transform([poutcome])[0]
    except ValueError as e:
        st.error(f"⚠️ Erreur d'encodage : {e}")
        st.stop()

    # Construction du vecteur — même ordre que X lors de l'entraînement
    # age, job, marital, education, housing, loan, contact,
    # month, day_of_week, duration, campaign, pdays, previous, poutcome
    X_input = np.array([[
        age, job_enc, marital_enc, education_enc,
        housing_enc, loan_enc, contact_enc,
        month_enc, day_enc, duration,
        campaign, pdays, previous, poutcome_enc
    ]])

    # Mise à l'échelle
    X_scaled = scaler.transform(X_input)

    # Prédiction
    prediction   = modele.predict(X_scaled)[0]
    probabilites = modele.predict_proba(X_scaled)[0]
    prob_no      = probabilites[0] * 100
    prob_yes     = probabilites[1] * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Résultat de la prédiction")

    if prediction == 1:
        st.success("✅ Ce client va probablement **SOUSCRIRE** au dépôt à terme")
        couleur = "#1a73e8"
        emoji   = "✅"
        label   = "SOUSCRIPTION — OUI"
    else:
        st.warning("❌ Ce client ne souscrira probablement **PAS** au dépôt à terme")
        couleur = "#e53935"
        emoji   = "❌"
        label   = "SOUSCRIPTION — NON"

    # Carte résultat
    st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {couleur}15, {couleur}30);
            border-left: 5px solid {couleur};
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
        '>
            <h2 style='color:{couleur}; margin:0'>{emoji} {label}</h2>
            <p style='margin:0.5rem 0 0 0; color:#333'>
                Confiance du modèle : <b>{max(prob_no, prob_yes):.1f}%</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Probabilités
    st.markdown("#### Probabilités détaillées")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("❌ Ne souscrira pas", f"{prob_no:.1f}%")
        st.progress(prob_no / 100)
    with col_b:
        st.metric("✅ Souscrira", f"{prob_yes:.1f}%")
        st.progress(prob_yes / 100)

    # Récapitulatif
    st.markdown("#### 📋 Récapitulatif client")
    recap = pd.DataFrame({
        "Variable": [
            "Âge", "Profession", "Statut marital", "Éducation",
            "Prêt immobilier", "Prêt personnel", "Contact",
            "Mois", "Jour", "Durée appel (sec)",
            "Nb contacts campagne", "Jours depuis dernier contact",
            "Nb contacts précédents", "Résultat campagne préc."
        ],
        "Valeur": [
            age, job, marital, education,
            housing, loan, contact,
            month, day_of_week, duration,
            campaign, pdays, previous, poutcome
        ]
    })
    st.dataframe(recap, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <div style='text-align:center; color:#999; font-size:0.85rem; padding:1rem 0'>
        Modèle : <b>Gradient Boosting</b> — Accuracy : <b>85.4%</b><br>
        Données : <a href='https://archive.ics.uci.edu/dataset/222/bank+marketing'
        target='_blank'>UCI Bank Marketing Dataset</a>
    </div>
""", unsafe_allow_html=True)