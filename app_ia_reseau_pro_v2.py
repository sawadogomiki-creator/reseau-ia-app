"""
Système IA de Prédiction de Défaillance des Transformateurs de Distribution
Version 3 — Parc de 3 transformateurs (cabine maçonnée / préfabriqué / haut de poteau)
Mode manuel (curseurs animés) + Mode simulation (pannes automatiques)
Carte du parc à Ouagadougou + prédiction avec prise en compte de la tendance (vitesse d'évolution du risque)
"""

import os
import time
import math
from collections import deque
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ============================================================================
# CONFIGURATION GÉNÉRALE
# ============================================================================

st.set_page_config(page_title="IA Réseau Pro — Parc SONABEL", page_icon="⚡", layout="wide")

MODEL_PATH = "modele_xgboost.pkl"
HISTORY_LEN = 60          # nombre de points d'historique conservés par transformateur
TICK_SECONDS = 1          # pas de l'horloge simulée
ANIMATION_STEP = 0.18     # vitesse de convergence des curseurs vers leur valeur cible (mode manuel)
OUAGA_LAT, OUAGA_LON = 12.3714, -1.5197

# --- Le parc : 3 transformateurs, un par type, situés dans 3 quartiers réels de Ouagadougou ---
TRANSFORMERS = [
    {
        "id": 1, "nom": "TR-01", "type": "Cabine maçonnée",
        "quartier": "Ouaga 2000", "lat": 12.3350, "lon": -1.4810,
        "obstacles": [
            "Immeubles administratifs proches (accès camion facile)",
            "Voie bitumée dégagée jusqu'au poste",
            "Peu de végétation à proximité immédiate",
        ],
        "vulnerabilite": "Faible exposition climatique directe ; bonne protection mécanique ; accès aisé pour la maintenance.",
    },
    {
        "id": 2, "nom": "TR-02", "type": "Préfabriqué",
        "quartier": "Gounghin", "lat": 12.3721, "lon": -1.5310,
        "obstacles": [
            "Marché de Gounghin à proximité (forte affluence, stationnement anarchique)",
            "Ligne aérienne basse tension croisant l'accès",
            "Caniveau d'évacuation des eaux pluviales longeant le poste",
        ],
        "vulnerabilite": "Exposition modérée ; accès parfois gêné par l'activité commerciale environnante.",
    },
    {
        "id": 3, "nom": "TR-03", "type": "Haut de poteau",
        "quartier": "Tanghin", "lat": 12.4010, "lon": -1.4870,
        "obstacles": [
            "Grand arbre à moins de 5 m (risque de chute de branches sur la ligne)",
            "Proximité du barrage de Tanghin (humidité ambiante élevée, berges)",
            "Habitations rapprochées limitant la manœuvre d'une nacelle",
        ],
        "vulnerabilite": "Forte exposition climatique (foudre, vent, humidité du barrage) ; accès à la nacelle parfois difficile.",
    },
]

# --- Modes de panne simulables automatiquement (au moins 5, cf. cahier des charges) ---
FAILURE_MODES = {
    "Surtension (foudre / manœuvre)": {
        "effet": {"voltage": +0.22, "oil_temp": +6, "charge": +0.05},
        "directives": [
            "Vérifier l'état des parafoudres et des mises à la terre du poste.",
            "Contrôler l'absence de traces d'amorçage sur les traversées.",
            "Programmer un essai de rigidité diélectrique de l'huile dans les 48 h.",
        ],
    },
    "Court-circuit interne": {
        "effet": {"charge": +0.35, "oil_temp": +18, "voltage": -0.15},
        "directives": [
            "Mettre hors tension immédiatement et consigner le transformateur.",
            "Réaliser une analyse des gaz dissous (DGA) avant toute remise en service.",
            "Inspecter le serrage des enroulements et l'état des connexions internes.",
        ],
    },
    "Température élevée / surcharge thermique": {
        "effet": {"oil_temp": +14, "ambient": +4, "charge": +0.15},
        "directives": [
            "Réduire la charge sur ce transformateur en reportant une partie des abonnés voisins.",
            "Nettoyer les radiateurs et vérifier l'absence d'encrassement par la poussière.",
            "Surveiller le thermomètre à cadran toutes les heures jusqu'à stabilisation.",
        ],
    },
    "Surcharge électrique prolongée": {
        "effet": {"charge": +0.40, "oil_temp": +10},
        "directives": [
            "Vérifier le taux de charge réel par rapport à la puissance nominale.",
            "Planifier un rééquilibrage des phases si un déséquilibre est constaté.",
            "Étudier un changement de transformateur pour une puissance supérieure si la surcharge est durable.",
        ],
    },
    "Défaut d'isolement (humidité)": {
        "effet": {"humidity": +25, "oil_temp": +5, "voltage": -0.08},
        "directives": [
            "Contrôler l'étanchéité du conservateur d'huile et l'état de l'assécheur d'air.",
            "Mesurer la teneur en eau de l'huile (méthode Karl Fischer) et la résistance d'isolement.",
            "Vérifier l'absence d'infiltration d'eau dans les boîtes de jonction basse tension.",
        ],
    },
}

RISK_BANDS = [(0, 35, "Faible", "🟢"), (35, 65, "Modéré", "🟠"), (65, 101, "Élevé", "🔴")]


def risk_band(pct: float):
    for lo, hi, label, emoji in RISK_BANDS:
        if lo <= pct < hi:
            return label, emoji
    return "Élevé", "🔴"


# ============================================================================
# CHARGEMENT DU MODÈLE (avec repli heuristique si le fichier est absent)
# ============================================================================

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.sidebar.warning(f"Modèle présent mais illisible ({e}) — repli sur le calcul heuristique.")
            return None
    return None


MODEL = load_model()

TYPE_MULTIPLIER = {"Cabine maçonnée": 0.85, "Préfabriqué": 1.0, "Haut de poteau": 1.20}


def predict_risk(features: dict, ttype: str) -> float:
    """Retourne un risque en pourcentage (0-100). Utilise le modèle XGBoost si
    disponible, sinon une formule heuristique cohérente avec les facteurs physiques
    du chapitre 1 (charge, température, humidité, tension, saison)."""
    season_map = {"Saison sèche fraîche": 0, "Saison sèche chaude": 1, "Harmattan": 2, "Saison des pluies": 3}
    season_code = season_map.get(features.get("season_label", "Saison sèche chaude"), 1)

    if MODEL is not None:
        try:
            x = pd.DataFrame([{
                "charge": features["charge"],
                "oil": features["oil_temp"],
                "ambient": features["ambient"],
                "humidity": features["humidity"],
                "voltage": features["voltage"],
                "season": season_code,
            }])
            proba = MODEL.predict_proba(x)[0][1]
            base = float(proba * 100)
        except Exception:
            base = _heuristic_risk(features, season_code)
    else:
        base = _heuristic_risk(features, season_code)

    return float(np.clip(base * TYPE_MULTIPLIER.get(ttype, 1.0), 0, 100))


def _heuristic_risk(f: dict, season_code: int) -> float:
    charge_term = max(0.0, f["charge"] - 0.8) * 55
    oil_term = max(0.0, f["oil_temp"] - 65) * 1.6
    ambient_term = max(0.0, f["ambient"] - 32) * 1.2
    humidity_term = max(0.0, f["humidity"] - 60) * 0.5
    voltage_term = abs(f["voltage"] - 1.0) * 60
    season_term = {0: 0, 1: 8, 2: 5, 3: 10}.get(season_code, 0)
    total = 8 + charge_term + oil_term + ambient_term + humidity_term + voltage_term + season_term
    return float(np.clip(total, 0, 100))


# ============================================================================
# MÉTÉO EN DIRECT (Ouagadougou) — via Open-Meteo, avec repli simulé
# ============================================================================

@st.cache_data(ttl=600)
def get_live_weather():
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={OUAGA_LAT}&longitude={OUAGA_LON}"
            "&current=temperature_2m,relative_humidity_2m&timezone=Africa%2FOuagadougou"
        )
        r = requests.get(url, timeout=4)
        data = r.json()
        temp = float(data["current"]["temperature_2m"])
        hum = float(data["current"]["relative_humidity_2m"])
        return temp, hum, True
    except Exception:
        month = datetime.now().month
        temp = 38 if month in (3, 4, 5) else (28 if month in (7, 8, 9) else 33)
        hum = 75 if month in (7, 8, 9) else 25
        return float(temp), float(hum), False


# ============================================================================
# ÉTAT DE SESSION
# ============================================================================

def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.running = True
    st.session_state.sim_time = datetime.now()
    st.session_state.history = {t["id"]: deque(maxlen=HISTORY_LEN) for t in TRANSFORMERS}

    # Mode manuel : valeur cible (curseur) et valeur affichée/effective (animée)
    st.session_state.manual_target = {
        t["id"]: {"voltage": 1.0, "charge": 0.8, "current_ratio": 0.8} for t in TRANSFORMERS
    }
    st.session_state.manual_effective = {
        t["id"]: {"voltage": 1.0, "charge": 0.8, "current_ratio": 0.8} for t in TRANSFORMERS
    }

    # Mode simulation : panne active par transformateur (ou None)
    st.session_state.sim_failure = {t["id"]: None for t in TRANSFORMERS}
    st.session_state.sim_elapsed = {t["id"]: 0 for t in TRANSFORMERS}


def animate_towards_target(t_id):
    tgt = st.session_state.manual_target[t_id]
    eff = st.session_state.manual_effective[t_id]
    for k in tgt:
        eff[k] += (tgt[k] - eff[k]) * ANIMATION_STEP
        if abs(eff[k] - tgt[k]) < 1e-3:
            eff[k] = tgt[k]


def push_history(t_id, risk_pct, features):
    st.session_state.history[t_id].append({
        "t": st.session_state.sim_time,
        "risk": risk_pct,
        **features,
    })


def compute_trend(t_id):
    """Calcule la pente du risque (points de %/minute) sur les dernières mesures,
    et une catégorie de tendance servant à amplifier l'attention portée à l'alerte."""
    hist = st.session_state.history[t_id]
    if len(hist) < 3:
        return 0.0, "stable"
    recent = list(hist)[-10:]
    t0 = recent[0]["t"]
    xs = [(p["t"] - t0).total_seconds() / 60.0 for p in recent]
    ys = [p["risk"] for p in recent]
    if xs[-1] - xs[0] < 1e-6:
        return 0.0, "stable"
    slope = np.polyfit(xs, ys, 1)[0]  # % par minute
    if slope < 0.5:
        cat = "stable"
    elif slope < 2.5:
        cat = "en hausse"
    elif slope < 6:
        cat = "en hausse rapide"
    else:
        cat = "critique — hausse brutale"
    return float(slope), cat


def attention_score(risk_pct, slope):
    """Combine le niveau absolu de risque et sa vitesse d'évolution : à niveau égal,
    un risque qui grimpe vite doit attirer davantage l'attention qu'un risque stable,
    même élevé depuis longtemps et déjà sous surveillance."""
    slope_bonus = np.clip(slope, 0, 10) * 4.0  # jusqu'à +40 pts si montée très rapide
    return float(np.clip(risk_pct * 0.75 + slope_bonus, 0, 100))


def prognosis_text(t_id, risk_pct, slope, failure_label=None):
    """Construit un pronostic du type :
    « Risque de court-circuit à venir : 40 % dans environ 5 h si rien n'est fait. »"""
    label = failure_label or "défaillance"
    if slope <= 0.15:
        return f"Évolution stable — pas d'aggravation significative détectée pour l'instant sur {label.lower()}."
    remaining_pct = max(0.0, 90 - risk_pct)
    minutes_to_90 = remaining_pct / slope if slope > 0 else None
    if minutes_to_90 is None or minutes_to_90 > 24 * 60:
        return f"Tendance à la hausse lente pour {label.lower()} — surveillance renforcée recommandée."
    hours = minutes_to_90 / 60
    if hours < 1:
        delay = f"environ {int(minutes_to_90)} min"
    else:
        delay = f"environ {hours:.1f} h"
    return (f"Risque de {label.lower()} à venir : {min(90, risk_pct + slope * 30):.0f} % "
            f"dans {delay} si rien n'est fait.")


# ============================================================================
# INTERFACE — EN-TÊTE, HORLOGE, MÉTÉO COMMUNE
# ============================================================================

init_state()

if HAS_AUTOREFRESH:
    st_autorefresh(interval=TICK_SECONDS * 1000, key="tick_refresh")

st.title("⚡ Système IA de Prédiction — Parc de transformateurs (Ouagadougou)")

top1, top2, top3 = st.columns([2, 2, 3])
with top1:
    if st.button("⏸️ Pause" if st.session_state.running else "▶️ Reprendre"):
        st.session_state.running = not st.session_state.running
with top2:
    st.metric("🕒 Horloge de simulation", st.session_state.sim_time.strftime("%d/%m/%Y %H:%M:%S"))
with top3:
    ambient_live, humidity_live, live_ok = get_live_weather()
    source = "en direct (Open-Meteo)" if live_ok else "estimation saisonnière (hors ligne)"
    st.metric(f"🌡️ Météo Ouagadougou — {source}", f"{ambient_live:.1f} °C / {humidity_live:.0f} % HR")

if st.session_state.running:
    st.session_state.sim_time += timedelta(seconds=TICK_SECONDS * 30)  # 30x accéléré

st.markdown(
    "Horloge en direct : le risque et son pourcentage sont recalculés à chaque tic pour "
    "chacun des 3 transformateurs du parc. Une alerte est mise en avant plus tôt lorsque "
    "le risque **grimpe vite**, même s'il n'a pas encore atteint le seuil critique."
)

with st.expander("📋 Étude comparative des 3 types de transformateurs du parc"):
    comp_df = pd.DataFrame([
        {"Type": t["type"], "Quartier": t["quartier"], "Vulnérabilité type": t["vulnerabilite"]}
        for t in TRANSFORMERS
    ])
    st.table(comp_df)

st.divider()

# ============================================================================
# NAVIGATION — 2 PARTIES
# ============================================================================

partie = st.radio(
    "Sélectionner le mode",
    ["🖐️ Partie 1 — Mode manuel", "🎲 Partie 2 — Mode simulation de pannes"],
    horizontal=True,
)

# ----------------------------------------------------------------------------
# PARTIE 1 — MODE MANUEL (curseurs individuels, effet progressif/animé)
# ----------------------------------------------------------------------------
if partie.startswith("🖐️"):
    st.subheader("Mode manuel — curseurs individuels par transformateur")
    st.caption("Une modification de curseur ne s'applique pas instantanément : la valeur "
               "effective glisse progressivement vers la valeur choisie, comme sur l'installation réelle.")

    cols = st.columns(3)
    for col, t in zip(cols, TRANSFORMERS):
        with col:
            st.markdown(f"### {t['nom']} — {t['type']}")
            st.caption(f"📍 {t['quartier']}")
            tgt = st.session_state.manual_target[t["id"]]
            tgt["voltage"] = st.slider("Tension (p.u.)", 0.85, 1.15, tgt["voltage"], 0.01, key=f"v_{t['id']}")
            tgt["current_ratio"] = st.slider("Courant relatif", 0.2, 1.6, tgt["current_ratio"], 0.02, key=f"c_{t['id']}")
            tgt["charge"] = st.slider("Charge relative", 0.2, 1.6, tgt["charge"], 0.02, key=f"ch_{t['id']}")

            animate_towards_target(t["id"])
            eff = st.session_state.manual_effective[t["id"]]

            oil_temp = 45 + eff["charge"] * 28 + max(0, ambient_live - 30) * 0.6
            features = {
                "charge": eff["charge"], "oil_temp": oil_temp, "ambient": ambient_live,
                "humidity": humidity_live, "voltage": eff["voltage"],
                "season_label": "Saison sèche chaude",
            }
            risk = predict_risk(features, t["type"])
            push_history(t["id"], risk, features)
            slope, trend_cat = compute_trend(t["id"])
            attn = attention_score(risk, slope)
            band_label, emoji = risk_band(attn)

            st.metric("Risque courant", f"{risk:.1f} %", delta=f"{slope:+.2f} pts/min")
            if trend_cat in ("en hausse rapide", "critique — hausse brutale"):
                st.error(f"{emoji} **{band_label} — {trend_cat.upper()}** : le risque augmente vite, priorité d'intervention.")
            elif band_label == "Élevé":
                st.error(f"{emoji} **Statut : {band_label}**")
            elif band_label == "Modéré":
                st.warning(f"{emoji} **Statut : {band_label}** ({trend_cat})")
            else:
                st.success(f"{emoji} **Statut : {band_label}** ({trend_cat})")

            hist = list(st.session_state.history[t["id"]])
            if len(hist) >= 2:
                dfh = pd.DataFrame(hist)
                fig = px.line(dfh, x="t", y="risk", title="Historique du risque")
                fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"hist_{t['id']}")

# ----------------------------------------------------------------------------
# PARTIE 2 — MODE SIMULATION (pannes automatiques, évolution simulée)
# ----------------------------------------------------------------------------
else:
    st.subheader("Mode simulation — déclenchement et évolution automatique de pannes")
    st.caption("Choisissez un mode de panne pour un ou plusieurs transformateurs : l'application "
               "fait évoluer progressivement les grandeurs électriques et thermiques, comme une "
               "dégradation réelle qui s'aggrave au fil du temps.")

    cols = st.columns(3)
    for col, t in zip(cols, TRANSFORMERS):
        with col:
            st.markdown(f"### {t['nom']} — {t['type']}")
            st.caption(f"📍 {t['quartier']}")

            options = ["Aucune (fonctionnement normal)"] + list(FAILURE_MODES.keys())
            current_choice = st.session_state.sim_failure[t["id"]] or options[0]
            choice = st.selectbox("Mode de panne simulé", options, index=options.index(current_choice)
                                   if current_choice in options else 0, key=f"fail_{t['id']}")

            if choice != st.session_state.sim_failure[t["id"]]:
                st.session_state.sim_failure[t["id"]] = None if choice == options[0] else choice
                st.session_state.sim_elapsed[t["id"]] = 0

            active_failure = st.session_state.sim_failure[t["id"]]

            base = {"charge": 0.8, "oil_temp": 60.0, "voltage": 1.0}
            if active_failure and st.session_state.running:
                st.session_state.sim_elapsed[t["id"]] += 1
            n = st.session_state.sim_elapsed[t["id"]]
            progress = 1 - math.exp(-n / 25.0)  # progression logarithmique de la dégradation

            if active_failure:
                effet = FAILURE_MODES[active_failure]["effet"]
                charge = base["charge"] + effet.get("charge", 0) * progress
                oil_temp = base["oil_temp"] + effet.get("oil_temp", 0) * progress
                voltage = base["voltage"] + effet.get("voltage", 0) * progress
                humidity_adj = humidity_live + effet.get("humidity", 0) * progress
                ambient_adj = ambient_live + effet.get("ambient", 0) * progress
            else:
                charge, oil_temp, voltage = base["charge"], base["oil_temp"], base["voltage"]
                humidity_adj, ambient_adj = humidity_live, ambient_live

            features = {
                "charge": charge, "oil_temp": oil_temp, "ambient": ambient_adj,
                "humidity": humidity_adj, "voltage": voltage,
                "season_label": "Saison sèche chaude",
            }
            risk = predict_risk(features, t["type"])
            push_history(t["id"], risk, features)
            slope, trend_cat = compute_trend(t["id"])
            attn = attention_score(risk, slope)
            band_label, emoji = risk_band(attn)

            st.metric("Risque courant", f"{risk:.1f} %", delta=f"{slope:+.2f} pts/min")

            if active_failure:
                if trend_cat in ("en hausse rapide", "critique — hausse brutale") or band_label == "Élevé":
                    st.error(f"{emoji} **{band_label} — {trend_cat}**")
                elif band_label == "Modéré":
                    st.warning(f"{emoji} **Statut : {band_label}** ({trend_cat})")
                else:
                    st.info(f"{emoji} Panne en cours de développement — statut encore {band_label.lower()}.")

                st.markdown(f"**Pronostic :** {prognosis_text(t['id'], risk, slope, active_failure)}")

                st.markdown("**Actions réseau recommandées :**")
                st.markdown("- Soulager le réseau en reportant une partie de la charge sur un poste voisin si possible.")
                st.markdown("- Vérifier et informer les clients connectés en aval de ce transformateur.")

                with st.expander("🩺 Diagnostic et directives correctives ciblées"):
                    for d in FAILURE_MODES[active_failure]["directives"]:
                        st.markdown(f"- {d}")
            else:
                st.success(f"{emoji} **Fonctionnement normal** ({trend_cat})")

            hist = list(st.session_state.history[t["id"]])
            if len(hist) >= 2:
                dfh = pd.DataFrame(hist)
                fig = px.line(dfh, x="t", y="risk", title="Historique du risque (simulation)")
                fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"simhist_{t['id']}")

st.divider()

# ============================================================================
# PARC NUMÉROTÉ + CARTE DU RÉSEAU
# ============================================================================

st.subheader("🗺️ Parc de transformateurs — localisation, risque et obstacles d'intervention")

rows = []
for t in TRANSFORMERS:
    hist = st.session_state.history[t["id"]]
    last_risk = hist[-1]["risk"] if hist else 0.0
    slope, trend_cat = compute_trend(t["id"])
    attn = attention_score(last_risk, slope)
    band_label, emoji = risk_band(attn)
    rows.append({
        "N°": t["id"], "Nom": t["nom"], "Type": t["type"], "Quartier": t["quartier"],
        "Risque (%)": round(last_risk, 1), "Tendance": trend_cat, "Statut": f"{emoji} {band_label}",
        "lat": t["lat"], "lon": t["lon"], "attn": attn,
    })

df_parc = pd.DataFrame(rows)
st.dataframe(df_parc.drop(columns=["lat", "lon", "attn"]), use_container_width=True, hide_index=True)

def color_for(attn):
    if attn >= 65:
        return [220, 40, 40, 200]
    if attn >= 35:
        return [240, 160, 30, 200]
    return [40, 160, 70, 200]

df_parc["color"] = df_parc["attn"].apply(color_for)
df_parc["radius"] = 120 + df_parc["attn"] * 3

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_parc,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius="radius",
    pickable=True,
)
view_state = pdk.ViewState(latitude=OUAGA_LAT, longitude=OUAGA_LON, zoom=11.5, pitch=0)
tooltip = {"text": "{Nom} ({Type}) — {Quartier}\nRisque : {Risque (%)} % — {Statut}"}
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip,
                          map_style="mapbox://styles/mapbox/light-v9"))

st.markdown("**Obstacles et contraintes d'accès relevés autour de chaque poste :**")
for t in TRANSFORMERS:
    with st.expander(f"{t['nom']} — {t['quartier']} ({t['type']})"):
        for o in t["obstacles"]:
            st.markdown(f"- {o}")

st.caption(
    "⚠️ Application de démonstration : les grandeurs électriques du mode manuel et les pannes du "
    "mode simulation sont générées par l'application ; la météo commune est récupérée en direct "
    "lorsque la connexion le permet, avec un repli saisonnier sinon."
)
