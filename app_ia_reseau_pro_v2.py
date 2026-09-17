import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="IA Réseau Pro", page_icon="⚡", layout="wide")

st.title("⚡ Système IA de Prédiction SAWADOGO")
st.markdown("Ce système utilise un modèle avancé **XGBoost** entraîné sur des lois de dégradation physique pour évaluer le risque de panne en temps réel.")

# 1. Chargement sécurisé du modèle XGBoost
MODEL_PATH = "modele_xgboost.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"⚠️ Le fichier '{MODEL_PATH}' est introuvable au même niveau que l'application sur GitHub.")
        return None

model = load_model()

# 2. Formulaire de saisie dans la barre latérale
st.sidebar.header("🔌 Paramètres Temps Réel")

charge = st.sidebar.slider("Charge relative du transformateur", 0.2, 1.5, 0.8, help="Ratio charge actuelle / charge nominale")
oil_temp = st.sidebar.slider("Température de l'huile (°C)", 30, 95, 65)
ambient_temp = st.sidebar.slider("Température ambiante (°C)", -5, 40, 20)
humidity = st.sidebar.slider("Humidité relative (%)", 10, 95, 50)
voltage = st.sidebar.slider("Tension relative (p.u.)", 0.85, 1.15, 1.0, help="Tension en unité relative (1.0 = nominal)")
season_label = st.sidebar.selectbox("Saison actuelle", ["Printemps", "Été", "Automne", "Hiver"])

# Correspondance exacte avec l'entraînement (0: Printemps, 1: Été, 2: Automne, 3: Hiver)
season_mapping = {"Printemps": 0, "Été": 1, "Automne": 2, "Hiver": 3}
season = season_mapping[season_label]

# Organisation de la page principale en deux colonnes
col1, col2 = st.columns([1, 1])

# 3. Calcul et Affichage du Risque (Colonne 1)
with col1:
    st.subheader("📊 Diagnostic de l'IA")
    
    if model is not None:
        # Préparation des données d'entrée au format exact attendu par XGBoost
        input_data = pd.DataFrame([{
            'charge': charge,
            'oil': oil_temp,
            'ambient': ambient_temp,
            'humidity': humidity,
            'voltage': voltage,
            'season': season
        }])
        
        # Calcul des probabilités de panne
        prediction_proba = model.predict_proba(input_data)[0][1]
        risk_percentage = float(prediction_proba * 100)
        
        # Affichage du score principal
        st.metric(label="Probabilité de défaillance critique", value=f"{risk_percentage:.2f} %")
        
        # Affichage du statut avec un code couleur clair
        if risk_percentage < 30:
            st.success("✅ **Statut : Réseau stable.** Le risque de panne est jugé faible. Maintenance de routine standard.")
        elif risk_percentage < 70:
            st.warning("⚠️ **Statut : Vigilance requise.** Risque modéré. Planifier une inspection préventive sur site.")
        else:
            st.error("🚨 **Statut Alerte : Risque très élevé !** Défaillance imminente probable. Déclencher une intervention d'urgence.")
            
    else:
        st.info("En attente de la détection du fichier de modèle...")

# 4. Graphique d'importance des variables (Colonne 2)
with col2:
    st.subheader("💡 Explication de la décision (XGBoost)")
    if model is not None:
        try:
            # Extraction des importances intrinsèques calculées par XGBoost lors de l'entraînement
            importances = model.feature_importances_
            features = ['charge', 'oil', 'ambient', 'humidity', 'voltage', 'season']
            
            # Création d'un tableau propre pour le graphique Plotly
            df_importance = pd.DataFrame({
                'Variable': ['Charge', 'Temp. Huile', 'Temp. Ambiante', 'Humidité', 'Tension', 'Saison'],
                'Importance (%)': importances * 100
            }).sort_values(by='Importance (%)', ascending=True)
            
            # Génération d'un graphique à barres horizontales épuré
            fig = px.bar(
                df_importance, 
                x='Importance (%)', 
                y='Variable', 
                orientation='h',
                title="Poids de chaque facteur dans le calcul du risque",
                color='Importance (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception:
            st.info("Le graphique d'importance sera disponible après la mise à jour complète de l'application.")
