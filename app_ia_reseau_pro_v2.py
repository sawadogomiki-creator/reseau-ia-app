"""
Système IA de Prédiction de Défaillance des Transformateurs de Distribution
Version 7.0 — Interface Ingénierie SCADA / SONABEL Control Room
"""

import os
import time
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import streamlit as st

# ============================================================================
# CONFIGURATION THÈME & INTERFACE SCADA
# ============================================================================
st.set_page_config(
    page_title="SONABEL - SCADA IA Réseau Pro", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection CSS pour forcer un rendu typé "Salle de Contrôle / Engineering"
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .stMetric { background-color: #1f2937; border-left: 4px solid #3b82f6; padding: 10px; border-radius: 4px; }
    div[data-testid="stBlock"] { padding: 5px; }
    .status-ok { color: #10b981; font-weight: bold; }
    .status-warn { color: #f59e0b; font-weight: bold; }
    .status-crit { color: #ef4444; font-weight: bold; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.3; } }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES & BASE DE DONNÉES TECHNIQUES
# ============================================================================
OUAGA_LAT, OUAGA_LON = 12.3714, -1.5197
FAILURE_TIME_CONSTANT = 60.0  # Accélération pour rendu ingénieur dynamique
HEAT_ALERT_THRESHOLD = 38.0

TRANSFORMERS = [
    {
        "id": 1, "nom": "TR-01/U2000", "type": "Cabine maçonnée", "puissance": "630 kVA",
        "quartier": "Ouaga 2000", "lat": 12.3350, "lon": -1.4810,
        "obstacles": ["Immeubles administratifs proches", "Voie bitumée dégagée"],
        "vulnerabilite": "Faible exposition climatique directe ; excellente isolation thermique passive.",
    },
    {
        "id": 2, "nom": "TR-02/GNGH", "type": "Préfabriqué", "puissance": "400 kVA",
        "quartier": "Gounghin", "lat": 12.3721, "lon": -1.5310,
        "obstacles": ["Marché à proximité (encombrement)", "Caniveau de drainage limitrophe"],
        "vulnerabilite": "Exposition thermique moyenne ; vibrations provoquées par le trafic lourd.",
    },
    {
        "id": 3, "nom": "TR-03/TNGH", "type": "Haut de poteau", "puissance": "160 kVA",
        "quartier": "Tanghin", "lat": 12.4010, "lon": -1.4870,
        "obstacles": ["Arbres de grand gabarit surplombant", "Humidité résiduelle du barrage"],
        "vulnerabilite": "Exposition critique (foudre, chocs thermiques directes, vents violents).",
    },
]

FAILURE_MODES = {
    "Surtension atmosphérique (Foudre)": {"v_mod": 1.9, "oil_inc": 4.5, "desc": "Amorçage de l'onde de choc foudre sur les traversées hautes tensions."},
    "Court-circuit interne enroulements": {"v_mod": 2.5, "oil_inc": 18.0, "desc": "Dégradation diélectrique majeure avec arc interne localisé."},
    "Surcharge thermique réseau": {"v_mod": 1.2, "oil_inc": 12.0, "desc": "Dépassement prolongé de la courbe de charge nominale (climatisation/pointe)."}
}

TYPE_TIME_FACTOR = {"Cabine maçonnée": 1.5, "Préfabriqué": 1.0, "Haut de poteau": 0.6}
WEATHER_PRESETS = {
    "🟢 Régime Standard (Ciel dégagé)": {"h_bonus": 0, "orage": False},
    "⚠️ Front Orageux (Pluie + Foudre)": {"h_bonus": 35, "orage": True},
    "🔥 Pic Caniculaire": {"h_bonus": -20, "orage": False}
}

# ============================================================================
# INITIALISATION ET MOTEUR TEMPS RÉEL (STATE)
# ============================================================================
if "current_tab" not in st.session_state: st.session_state.current_tab = "📟 Console SCADA"
if "simulation_time" not in st.session_state: st.session_state.simulation_time = datetime.now()
if "base_temp" not in st.session_state: st.session_state.base_temp = 36.5
if "weather_preset" not in st.session_state: st.session_state.weather_preset = "🟢 Régime Standard (Ciel dégagé)"
if "historique_pannes" not in st.session_state: st.session_state.historique_pannes = []
if "telemetrie_history" not in st.session_state: st.session_state.telemetrie_history = deque(maxlen=20)

for tr in TRANSFORMERS:
    tid = tr["id"]
    if f"tr_{tid}_progression" not in st.session_state: st.session_state[f"tr_{tid}_progression"] = 0.0
    if f"tr_{tid}_panne_active" not in st.session_state: st.session_state[f"tr_{tid}_panne_active"] = None

# ---- Calculs itératifs de simulation ----
st.session_state.simulation_time += timedelta(seconds=20)
preset_data = WEATHER_PRESETS[st.session_state.weather_preset]
current_humidity = max(5, min(98, 40 + preset_data["h_bonus"]))

snapshot = {"time": st.session_state.simulation_time.strftime("%H:%M:%S")}
for tr in TRANSFORMERS:
    tid = tr["id"]
    panne = st.session_state[f"tr_{tid}_panne_active"]
    
    if panne:
        vitesse = (1.0 / (FAILURE_TIME_CONSTANT * TYPE_TIME_FACTOR[tr["type"]])) * FAILURE_MODES[panne]["v_mod"]
        if st.session_state.base_temp > HEAT_ALERT_THRESHOLD: vitesse *= 1.3
        
        prev_prog = st.session_state[f"tr_{tid}_progression"]
        st.session_state[f"tr_{tid}_progression"] = min(1.0, prev_prog + vitesse)
        
        if st.session_state[f"tr_{tid}_progression"] == 1.0 and prev_prog < 1.0:
            st.session_state.historique_pannes.append({
                "Horodatage": st.session_state.simulation_time.strftime("%H:%M:%S"),
                "Code Équipement": tr["nom"],
                "Type Panne": panne,
                "Gravité": "CRITIQUE (00% Vie)"
            })
    else:
        st.session_state[f"tr_{tid}_progression"] = max(0.0, st.session_state[f"tr_{tid}_progression"] - 0.02)

    snapshot[tr["nom"]] = (1.0 - st.session_state[f"tr_{tid}_progression"]) * 100
st.session_state.telemetrie_history.append(snapshot)

# ============================================================================
# SIDEBAR RE-DESIGNÉE (PANNEAU DE CONFIGURATION)
# ============================================================================
with st.sidebar:
    st.markdown("### ⚡ **SONABEL COCKPIT v7**")
    st.markdown(f"**Statut Système :** Online <span class='status-ok'>●</span>", unsafe_allow_html=True)
    st.write("---")
    
    st.metric(label="🛰️ HORLOGE DE BORD (MMS)", value=st.session_state.simulation_time.strftime("%H:%M:%S"), delta=st.session_state.simulation_time.strftime("%d/%m/%Y"))
    st.write("---")
    
    st.subheader("🎛️ Navigation Matrice")
    if st.button("📟 Console SCADA (Principal)", use_container_width=True): st.session_state.current_tab = "📟 Console SCADA"
    if st.button("🗺️ Synoptique SIG (Carte)", use_container_width=True): st.session_state.current_tab = "🗺️ Synoptique SIG"
    if st.button("🩺 Diagnostics & Abaques", use_container_width=True): st.session_state.current_tab = "🩺 Diagnostics & Abaques"
    if st.button("📊 Archives & Prévisions", use_container_width=True): st.session_state.current_tab = "📊 Archives & Prévisions"
    
    st.write("---")
    st.subheader("⚙️ Simulateur de Profils")
    st.session_state.base_temp = st.slider("Température Cellule (°C)", 20.0, 50.0, float(st.session_state.base_temp), step=0.5)
    st.session_state.weather_preset = st.selectbox("Vecteur Météo", list(WEATHER_PRESETS.keys()))
    
    st.write("---")
    st.markdown("⚠️ **Injection de Contraintes Avancées :**")
    inject_target = st.selectbox("Cible Réseau", [t["nom"] for t in TRANSFORMERS])
    inject_fault = st.selectbox("Vecteur Défaut", ["Aucun incident Actif"] + list(FAILURE_MODES.keys()))
    
    if st.button("⚡ Injecter l'état dans la boucle de calcul", use_container_width=True):
        t_obj = next(t for t in TRANSFORMERS if t["nom"] == inject_target)
        if "Aucun" in inject_fault:
            st.session_state[f"tr_{t_obj['id']}_panne_active"] = None
            st.session_state[f"tr_{t_obj['id']}_progression"] = 0.0
        else:
            st.session_state[f"tr_{t_obj['id']}_panne_active"] = inject_fault
        st.rerun()

# ============================================================================
# ONGLER 1 : CONSOLE SCADA
# ============================================================================
if st.session_state.current_tab == "📟 Console SCADA":
    st.title("📟 Tableau de Bord d'Ingénierie Réseau")
    
    # KPIs Industriels supérieurs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    actifs_alarme = sum(1 for t in TRANSFORMERS if st.session_state[f"tr_{t['id']}_panne_active"] is not None)
    
    kpi1.metric("UNITÉS EN ANOMALIE", f"{actifs_alarme} / {len(TRANSFORMERS)}", delta="ANORMAL" if actifs_alarme>0 else "NOMINAL", delta_color="inverse")
    kpi2.metric("TEMPÉRATURE AMBIANTE", f"{st.session_state.base_temp} °C", "Seuil Critique: 38°C")
    kpi3.metric("HUMIDITÉ DU VECTEUR", f"{current_humidity} %", "Fluide gazeux")
    kpi4.metric("FRÉQUENCE RÉSEAU ESTIMÉE", "50.02 Hz", "± 0.04 Hz")
    
    st.write("---")
    st.subheader("💡 État Matriciel des Transformateurs de Puissance")
    
    for tr in TRANSFORMERS:
        tid = tr["id"]
        prog_risque = st.session_state[f"tr_{tid}_progression"] * 100
        panne = st.session_state[f"tr_{tid}_panne_active"]
        
        # Calcul de variables dérivés d'ingénierie
        base_temp_huile = 50.0 + (st.session_state.base_temp * 0.4)
        temp_huile_finale = base_temp_huile + (prog_risque * 0.6)
        charge_est = 45.0 + (prog_risque * 0.5) if panne else 62.4
        
        col_specs, col_telemetry, col_actions = st.columns([2, 3, 1])
        
        with col_specs:
            st.markdown(f"##### 🛡️ **{tr['nom']}** | `{tr['puissance']}`")
            st.caption(f"Architecture : {tr['type']} — Zone : {tr['quartier']}")
            if panne:
