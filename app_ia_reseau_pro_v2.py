"""
Système IA de Prédiction de Défaillance des Transformateurs de Distribution
Version 5 — Navigation par sections (boutons), horloge temps réel (comme une montre),
progression réaliste des pannes différenciée par type de transformateur,
actions réseau spécifiques par panne ET par type de poste, historique des pannes,
météo simulable (température + type de temps) avec impact différencié selon le type
de transformateur, carte réaliste (fond routier), et prévisions météo sur 7 jours.
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

# ============================================================================
# CONFIGURATION GÉNÉRALE
# ============================================================================

st.set_page_config(page_title="IA Réseau Pro — Parc SONABEL", page_icon="⚡", layout="wide")

MODEL_PATH = "modele_xgboost.pkl"
HISTORY_LEN = 120
ANIMATION_STEP = 0.18
OUAGA_LAT, OUAGA_LON = 12.3714, -1.5197
FAILURE_TIME_CONSTANT = 90.0   # secondes — vitesse d'installation d'une panne simulée (poste de référence)
HEAT_ALERT_THRESHOLD = 38.0    # °C — seuil utilisé pour l'indice de risque de forte chaleur

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

# Effet final (à progression = 1.0) de chaque mode de panne, atteint progressivement
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

# Libellé court utilisé pour l'affichage du risque nommé (mode manuel)
SHORT_FAILURE_LABELS = {
    "Surtension (foudre / manœuvre)": "surtension",
    "Court-circuit interne": "court-circuit",
    "Température élevée / surcharge thermique": "surchauffe thermique",
    "Surcharge électrique prolongée": "surcharge électrique",
    "Défaut d'isolement (humidité)": "défaut d'isolement (humidité)",
}

# Vitesse relative de progression d'une panne selon le type de poste : un transformateur
# sur poteau, totalement exposé, bascule plus vite vers l'état critique qu'une cabine
# maçonnée mieux protégée, à panne strictement identique.
TYPE_TIME_FACTOR = {"Cabine maçonnée": 1.35, "Préfabriqué": 1.0, "Haut de poteau": 0.7}

# Actions réseau recommandées : dépendent du type de panne...
NETWORK_ACTIONS_BY_FAILURE = {
    "Surtension (foudre / manœuvre)": [
        "Vérifier les autres postes du même départ HTA, susceptibles d'avoir subi la même surtension.",
        "Basculer temporairement les abonnés sensibles sur un départ voisin si des perturbations sont constatées.",
    ],
    "Court-circuit interne": [
        "Isoler immédiatement ce transformateur du réseau et consigner le disjoncteur amont.",
        "Réalimenter les abonnés concernés depuis un poste de secours ou un départ voisin dans les meilleurs délais.",
    ],
    "Température élevée / surcharge thermique": [
        "Reporter une partie de la charge de ce transformateur vers un poste voisin le temps du refroidissement.",
        "Limiter temporairement les nouveaux branchements (climatisation, groupes de secours) sur ce départ.",
    ],
    "Surcharge électrique prolongée": [
        "Étaler dans le temps les nouveaux raccordements prévus sur ce transformateur.",
        "Étudier un transfert durable d'une partie des abonnés vers un poste moins chargé.",
    ],
    "Défaut d'isolement (humidité)": [
        "Éviter toute manœuvre sous tension tant que l'isolement n'est pas rétabli.",
        "Prévoir une intervention hors période pluvieuse pour l'entretien du conservateur d'huile.",
    ],
}

# ...et du type de poste (contraintes d'accès et d'intervention propres à chaque site).
NETWORK_ACTIONS_BY_TYPE = {
    "Cabine maçonnée": "Accès véhicule dégagé : une équipe peut intervenir rapidement, sans contrainte majeure.",
    "Préfabriqué": "Prévenir les commerçants et usagers à proximité avant toute manœuvre : l'accès peut être encombré.",
    "Haut de poteau": "Intervention en hauteur : mobiliser une nacelle et sécuriser le dégagement autour du poteau avant toute coupure.",
}

# Conditions météo simulables dans la section « Simulation de pannes » : bonus/malus
# d'humidité relative appliqué à la météo de référence, et déclenchement ou non du
# risque de foudre associé à l'orage.
WEATHER_PRESETS = {
    "☀️ Ciel dégagé (normal)": {"humidity_bonus": 0, "orage": False},
    "🌧️ Pluie": {"humidity_bonus": 30, "orage": False},
    "⛈️ Orage (pluie + foudre)": {"humidity_bonus": 35, "orage": True},
    "🌬️ Fraîcheur / harmattan frais": {"humidity_bonus": -15, "orage": False},
    "🔥 Canicule": {"humidity_bonus": -10, "orage": False},
}

# Sensibilité de chaque type de transformateur au climat ambiant : un poteau, exposé
# directement à l'air libre et à la pluie, ressent bien plus l'effet de la chaleur, de
# l'humidité et de la foudre qu'une cabine maçonnée qui protège mécaniquement l'appareil.
TYPE_CLIMATE = {
    "Cabine maçonnée": {"ambient_factor": 0.5, "humidity_factor": 0.35, "orage_voltage_bonus": 0.02},
    "Préfabriqué": {"ambient_factor": 0.85, "humidity_factor": 0.7, "orage_voltage_bonus": 0.05},
    "Haut de poteau": {"ambient_factor": 1.2, "humidity_factor": 1.0, "orage_voltage_bonus": 0.10},
}

RISK_BANDS = [
    (0, 35, "Faible", "🟢"),
    (35, 50, "Modéré", "🟠"),
    (50, 65, "Vigilance avancée", "🟡"),
    (65, 101, "Élevé", "🔴"),
]
SECTIONS = ["🖐️ Mode manuel", "🎲 Simulation de pannes", "📋 État des transformateurs",
            "📈 Historique des pannes", "🌦️ Météo & prévisions", "🗺️ Carte du parc"]


def risk_band(pct: float):
    for lo, hi, label, emoji in RISK_BANDS:
        if lo <= pct < hi:
            return label, emoji
    return "Élevé", "🔴"


# ============================================================================
# MODÈLE DE PRÉDICTION (XGBoost si disponible, sinon heuristique)
# ============================================================================

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


MODEL = load_model()
TYPE_MULTIPLIER = {"Cabine maçonnée": 0.85, "Préfabriqué": 1.0, "Haut de poteau": 1.20}


def predict_risk(features: dict, ttype: str) -> float:
    season_map = {"Saison sèche fraîche": 0, "Saison sèche chaude": 1, "Harmattan": 2, "Saison des pluies": 3}
    season_code = season_map.get(features.get("season_label", "Saison sèche chaude"), 1)
    if MODEL is not None:
        try:
            x = pd.DataFrame([{
                "charge": features["charge"], "oil": features["oil_temp"],
                "ambient": features["ambient"], "humidity": features["humidity"],
                "voltage": features["voltage"], "season": season_code,
            }])
            base = float(MODEL.predict_proba(x)[0][1] * 100)
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


def infer_dominant_failure(features: dict, baseline: dict):
    """Détermine, à partir de l'écart entre les valeurs courantes et une situation
    normale de référence, quelle panne (parmi FAILURE_MODES) se rapproche le plus de
    l'état observé. Utilisé en mode manuel pour nommer le risque affiché."""
    scores = {}
    for label, mode in FAILURE_MODES.items():
        effet = mode["effet"]
        contrib, weight = 0.0, 0.0
        for key, delta in effet.items():
            if key not in features or key not in baseline or delta == 0:
                continue
            diff = features[key] - baseline[key]
            contrib += diff / delta  # >0 si l'écart va dans le sens de cette panne
            weight += 1.0
        if weight > 0:
            scores[label] = contrib / weight
    if not scores:
        return None, 0.0
    best_label = max(scores, key=scores.get)
    return best_label, scores[best_label]


# ============================================================================
# MÉTÉO — direct + prévisions 7 jours
# ============================================================================

@st.cache_data(ttl=600)
def get_live_weather():
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={OUAGA_LAT}&longitude={OUAGA_LON}"
               "&current=temperature_2m,relative_humidity_2m&timezone=Africa%2FOuagadougou")
        data = requests.get(url, timeout=4).json()
        return float(data["current"]["temperature_2m"]), float(data["current"]["relative_humidity_2m"]), True
    except Exception:
        month = datetime.now().month
        temp = 38 if month in (3, 4, 5) else (28 if month in (7, 8, 9) else 33)
        hum = 75 if month in (7, 8, 9) else 25
        return float(temp), float(hum), False


@st.cache_data(ttl=1800)
def get_weekly_forecast():
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={OUAGA_LAT}&longitude={OUAGA_LON}"
               "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
               "&forecast_days=7&timezone=Africa%2FOuagadougou")
        data = requests.get(url, timeout=5).json()
        d = data["daily"]
        rows = []
        for i, date in enumerate(d["time"]):
            tmax = float(d["temperature_2m_max"][i])
            tmin = float(d["temperature_2m_min"][i])
            rain_pct = float(d["precipitation_probability_max"][i])
            heat_pct = float(np.clip((tmax - 32) / 13 * 100, 0, 100))
            rows.append({"date": date, "tmax": tmax, "tmin": tmin,
                         "pluie_pct": rain_pct, "chaleur_pct": heat_pct})
        return rows, True
    except Exception:
        month = datetime.now().month
        rainy = month in (6, 7, 8, 9, 10)
        rows = []
        for i in range(7):
            d = datetime.now() + timedelta(days=i)
            tmax = 36 + np.random.uniform(-3, 4) if not rainy else 31 + np.random.uniform(-2, 3)
            rain_pct = np.random.uniform(40, 85) if rainy else np.random.uniform(0, 15)
            heat_pct = float(np.clip((tmax - 32) / 13 * 100, 0, 100))
            rows.append({"date": d.strftime("%Y-%m-%d"), "tmax": round(tmax, 1),
                         "tmin": round(tmax - 8, 1), "pluie_pct": round(rain_pct, 0),
                         "chaleur_pct": round(heat_pct, 0)})
        return rows, False


# ============================================================================
# ÉTAT DE SESSION
# ============================================================================

def init_state():
    """Initialise chaque clé de session individuellement (plutôt qu'un seul drapeau global) :
    si une session existante (ouverte avant une mise à jour du code) n'a pas encore telle ou
    telle clé, elle est comblée ici sans jamais écraser une valeur déjà en cours d'utilisation."""
    defaults = {
        "running": True,
        "section": SECTIONS[0],
        "history": lambda: {t["id"]: deque(maxlen=HISTORY_LEN) for t in TRANSFORMERS},
        "manual_target": lambda: {t["id"]: {"voltage": 1.0, "charge": 0.8, "current_ratio": 0.8} for t in TRANSFORMERS},
        "manual_effective": lambda: {t["id"]: {"voltage": 1.0, "charge": 0.8, "current_ratio": 0.8} for t in TRANSFORMERS},
        "sim_failure": lambda: {t["id"]: None for t in TRANSFORMERS},
        "sim_start_time": lambda: {t["id"]: None for t in TRANSFORMERS},
        "event_log": list,
        "active_event": lambda: {t["id"]: None for t in TRANSFORMERS},
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default() if callable(default) else default

    # Filet de sécurité supplémentaire : si TRANSFORMERS évolue (ajout d'un poste) alors
    # qu'une session est déjà ouverte, on complète les sous-dictionnaires par transformateur
    # plutôt que de laisser un KeyError se produire lors d'un accès à un identifiant absent.
    per_transfo_defaults = {
        "manual_target": {"voltage": 1.0, "charge": 0.8, "current_ratio": 0.8},
        "manual_effective": {"voltage": 1.0, "charge": 0.8, "current_ratio": 0.8},
        "sim_failure": None,
        "sim_start_time": None,
        "active_event": None,
    }
    for key, default_value in per_transfo_defaults.items():
        for t in TRANSFORMERS:
            st.session_state[key].setdefault(
                t["id"], dict(default_value) if isinstance(default_value, dict) else default_value
            )
    for t in TRANSFORMERS:
        if t["id"] not in st.session_state.history:
            st.session_state.history[t["id"]] = deque(maxlen=HISTORY_LEN)


def animate_towards_target(t_id):
    tgt = st.session_state.manual_target[t_id]
    eff = st.session_state.manual_effective[t_id]
    for k in tgt:
        eff[k] += (tgt[k] - eff[k]) * ANIMATION_STEP
        if abs(eff[k] - tgt[k]) < 1e-3:
            eff[k] = tgt[k]


def push_history(t_id, risk_pct, features):
    st.session_state.history[t_id].append({"t": datetime.now(), "risk": risk_pct, **features})


def compute_trend(t_id):
    hist = st.session_state.history[t_id]
    if len(hist) < 3:
        return 0.0, "stable"
    recent = list(hist)[-12:]
    t0 = recent[0]["t"]
    xs = [(p["t"] - t0).total_seconds() / 60.0 for p in recent]
    ys = [p["risk"] for p in recent]
    if xs[-1] - xs[0] < 1e-6:
        return 0.0, "stable"
    slope = np.polyfit(xs, ys, 1)[0]
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
    slope_bonus = np.clip(slope, 0, 10) * 4.0
    return float(np.clip(risk_pct * 0.75 + slope_bonus, 0, 100))


def prognosis_text(risk_pct, slope, failure_label=None):
    label = failure_label or "défaillance"
    if slope <= 0.15:
        return f"Évolution stable — pas d'aggravation significative détectée pour {label.lower()}."
    remaining_pct = max(0.0, 90 - risk_pct)
    minutes_to_90 = remaining_pct / slope if slope > 0 else None
    if minutes_to_90 is None or minutes_to_90 > 24 * 60:
        return f"Tendance à la hausse lente pour {label.lower()} — surveillance renforcée recommandée."
    hours = minutes_to_90 / 60
    delay = f"environ {int(minutes_to_90)} min" if hours < 1 else f"environ {hours:.1f} h"
    return (f"Risque de {label.lower()} à venir : {min(90, risk_pct + slope * 30):.0f} % "
            f"dans {delay} si rien n'est fait.")


def failure_phase(progress: float):
    if progress < 0.12:
        return "🔍 Anomalie naissante", "info", (
            "Un écart encore ténu vient d'apparaître sur ce transformateur. Rien de visible "
            "à l'œil nu, mais le modèle détecte déjà une dérive et la place sous surveillance."
        )
    if progress < 0.45:
        return "⚠️ Dégradation en cours", "warning", (
            "L'écart se creuse nettement : les grandeurs électriques et thermiques s'éloignent "
            "progressivement de leur plage normale. Une intervention préventive limiterait l'aggravation."
        )
    if progress < 0.8:
        return "🟠 Aggravation avancée", "warning", (
            "La dégradation s'accélère. Sans action, l'état critique sera atteint dans un délai court."
        )
    return "🚨 État critique", "error", (
        "Le transformateur a atteint un état proche de la défaillance complète. Intervention urgente requise."
    )


def log_event_if_needed(t_id, nom, band_label, risk, panne_label):
    active = st.session_state.active_event[t_id]
    if band_label == "Élevé":
        if active is None:
            st.session_state.event_log.append({
                "transfo_id": t_id, "nom": nom, "type_panne": panne_label or "Mode manuel",
                "debut": datetime.now(), "fin": None, "risque_max": risk,
            })
            st.session_state.active_event[t_id] = len(st.session_state.event_log) - 1
        else:
            ev = st.session_state.event_log[active]
            ev["risque_max"] = max(ev["risque_max"], risk)
    else:
        if active is not None:
            st.session_state.event_log[active]["fin"] = datetime.now()
            st.session_state.active_event[t_id] = None


def get_failure_progress(t_id, ttype):
    """Lecture (sans effet de bord) de la panne simulée en cours pour ce transformateur,
    utilisée par la section « État des transformateurs » pour refléter l'état actuel de
    la simulation, quelle que soit la section affichée à l'écran."""
    active = st.session_state.sim_failure.get(t_id)
    if not active:
        return None, 0.0
    start = st.session_state.sim_start_time.get(t_id)
    if start is None:
        return active, 0.0
    elapsed = max(0.0, time.time() - start)
    effective_time_constant = FAILURE_TIME_CONSTANT * TYPE_TIME_FACTOR.get(ttype, 1.0)
    progress = 1 - math.exp(-elapsed / effective_time_constant)
    return active, progress


# ============================================================================
# INITIALISATION + EN-TÊTE (horloge « comme une montre » + météo commune)
# ============================================================================

init_state()

st.title("⚡ Système IA de Prédiction — Parc de transformateurs (Ouagadougou)")

ambient_live, humidity_live, live_ok = get_live_weather()

top1, top2, top3 = st.columns([2, 2, 3])
with top1:
    if st.button("⏸️ Pause" if st.session_state.running else "▶️ Reprendre", key="run_toggle"):
        st.session_state.running = not st.session_state.running
with top2:
    st.metric("🕒 Heure actuelle", datetime.now().strftime("%H:%M:%S"))
    st.caption(datetime.now().strftime("%A %d %B %Y"))
with top3:
    source = "en direct (Open-Meteo)" if live_ok else "estimation saisonnière (hors ligne)"
    st.metric(f"🌡️ Météo Ouagadougou — {source}", f"{ambient_live:.1f} °C / {humidity_live:.0f} % HR")

st.caption(
    "L'horloge affiche l'heure réelle, comme une montre. Tant que la simulation n'est pas en "
    "pause, la page se rafraîchit chaque seconde et les pannes en cours poursuivent leur évolution."
)

with st.expander("📋 Étude comparative des 3 types de transformateurs du parc"):
    st.table(pd.DataFrame([
        {"Type": t["type"], "Quartier": t["quartier"], "Vulnérabilité type": t["vulnerabilite"]}
        for t in TRANSFORMERS
    ]))

st.divider()

# ============================================================================
# NAVIGATION PAR BOUTONS
# ============================================================================

nav_cols = st.columns(len(SECTIONS))
for col, sec in zip(nav_cols, SECTIONS):
    is_active = st.session_state.section == sec
    if col.button(sec, key=f"nav_{sec}", use_container_width=True,
                  type="primary" if is_active else "secondary"):
        st.session_state.section = sec

st.divider()
section = st.session_state.section

# ============================================================================
# SECTION — MODE MANUEL
# ============================================================================
if section == "🖐️ Mode manuel":
    st.subheader("Mode manuel — curseurs individuels par transformateur")
    st.caption("Une modification de curseur ne s'applique pas instantanément : la valeur "
               "effective glisse progressivement vers la valeur choisie. Le risque affiché "
               "est nommé selon le type de panne auquel la situation courante se rapproche le plus.")

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
            features = {"charge": eff["charge"], "oil_temp": oil_temp, "ambient": ambient_live,
                        "humidity": humidity_live, "voltage": eff["voltage"], "season_label": "Saison sèche chaude"}
            risk = predict_risk(features, t["type"])
            push_history(t["id"], risk, features)
            slope, trend_cat = compute_trend(t["id"])
            attn = attention_score(risk, slope)
            band_label, emoji = risk_band(attn)

            # Nom de la panne dominante à laquelle la situation courante ressemble le plus,
            # calculé par écart à une situation de référence (charge/tension/T° huile normales).
            baseline = {
                "voltage": 1.0, "charge": 0.8,
                "oil_temp": 45 + 0.8 * 28 + max(0, ambient_live - 30) * 0.6,
                "humidity": humidity_live,
            }
            dominant_label, dominant_score = infer_dominant_failure(features, baseline)
            if dominant_label and dominant_score > 0.15:
                risk_title = f"Risque de {SHORT_FAILURE_LABELS[dominant_label]}"
                log_event_if_needed(t["id"], t["nom"], band_label, risk, dominant_label)
            else:
                risk_title = "Risque global (aucune anomalie dominante)"
                log_event_if_needed(t["id"], t["nom"], band_label, risk, None)

            st.metric(risk_title, f"{risk:.1f} %", delta=f"{slope:+.2f} pts/min")
            if trend_cat in ("en hausse rapide", "critique — hausse brutale"):
                st.error(f"{emoji} **{band_label} — {trend_cat.upper()}** : priorité d'intervention.")
            elif band_label == "Élevé":
                st.error(f"{emoji} **Statut : {band_label}**")
            elif band_label == "Vigilance avancée":
                st.warning(f"{emoji} **Statut : {band_label}** — surveillance renforcée recommandée ({trend_cat})")
            elif band_label == "Modéré":
                st.warning(f"{emoji} **Statut : {band_label}** ({trend_cat})")
            else:
                st.success(f"{emoji} **Statut : {band_label}** ({trend_cat})")

# ============================================================================
# SECTION — SIMULATION DE PANNES (progression différenciée + météo simulable)
# ============================================================================
elif section == "🎲 Simulation de pannes":
    st.subheader("Mode simulation — du fonctionnement normal à la panne, progressivement")
    st.caption("Le modèle détecte une dérive dès les premières secondes, bien avant que le "
               "risque ne devienne critique. À panne identique, la vitesse d'aggravation et les "
               "actions recommandées diffèrent selon le type de poste (cabine, préfabriqué, poteau).")

    st.markdown("#### 🌦️ Conditions climatiques simulées (communes aux 3 postes)")
    wc1, wc2 = st.columns(2)
    with wc1:
        sim_temp = st.slider("Température ambiante simulée (°C)", 15, 50,
                              int(round(np.clip(ambient_live, 15, 50))), 1, key="sim_temp_slider")
    with wc2:
        sim_weather_label = st.selectbox("Conditions météo simulées", list(WEATHER_PRESETS.keys()),
                                          key="sim_weather_select")
    weather = WEATHER_PRESETS[sim_weather_label]
    st.caption(
        "La température et le type de temps choisis ici remplacent la météo en direct le temps de "
        "la simulation. Leur impact réel sur le transformateur (échauffement interne, humidité "
        "perçue, risque de coup de foudre) dépend ensuite du type de poste : un poteau, exposé à "
        "l'air libre, ressent le climat bien plus fortement qu'une cabine maçonnée protégée."
    )

    cols = st.columns(3)
    for col, t in zip(cols, TRANSFORMERS):
        with col:
            st.markdown(f"### {t['nom']} — {t['type']}")
            st.caption(f"📍 {t['quartier']}")

            options = ["Aucune (fonctionnement normal)"] + list(FAILURE_MODES.keys())
            current_choice = st.session_state.sim_failure.get(t["id"]) or options[0]
            choice = st.selectbox("Mode de panne simulé", options,
                                   index=options.index(current_choice) if current_choice in options else 0,
                                   key=f"fail_{t['id']}")

            if choice != (st.session_state.sim_failure.get(t["id"]) or options[0]):
                if choice == options[0]:
                    st.session_state.sim_failure[t["id"]] = None
                    st.session_state.sim_start_time[t["id"]] = None
                else:
                    st.session_state.sim_failure[t["id"]] = choice
                    st.session_state.sim_start_time[t["id"]] = time.time()

            active_failure = st.session_state.sim_failure.get(t["id"])
            climate = TYPE_CLIMATE.get(t["type"], {"ambient_factor": 1.0, "humidity_factor": 1.0, "orage_voltage_bonus": 0.0})

            # Impact du climat simulé sur ce type de poste précisément : la chaleur ambiante ne
            # se traduit pas de la même façon en T° huile selon l'exposition, l'humidité ressentie
            # varie aussi, et un orage induit une surtension d'autant plus marquée que le poste
            # est exposé (poteau) plutôt que protégé (cabine maçonnée).
            oil_temp_climate_add = max(0.0, sim_temp - 30) * 0.6 * climate["ambient_factor"]
            humidity_climate = float(np.clip(humidity_live + weather["humidity_bonus"] * climate["humidity_factor"], 5, 100))
            voltage_climate_add = climate["orage_voltage_bonus"] if weather["orage"] else 0.0

            base = {
                "charge": 0.8,
                "oil_temp": 45 + 0.8 * 28 + oil_temp_climate_add,
                "voltage": 1.0 + voltage_climate_add,
            }

            if active_failure:
                start = st.session_state.sim_start_time.get(t["id"])
                if start is None:
                    # Filet de sécurité : une panne est marquée active mais son horodatage de
                    # départ est manquant (état incohérent) — on redémarre le chrono maintenant
                    # plutôt que de planter sur une soustraction avec None.
                    start = time.time()
                    st.session_state.sim_start_time[t["id"]] = start
                elapsed = max(0.0, time.time() - start)
                # Constante de temps propre au type de poste : à panne strictement identique,
                # un poteau exposé atteint l'état critique plus vite qu'une cabine protégée.
                effective_time_constant = FAILURE_TIME_CONSTANT * TYPE_TIME_FACTOR.get(t["type"], 1.0)
                progress = 1 - math.exp(-elapsed / effective_time_constant)
                effet = FAILURE_MODES[active_failure]["effet"]
                charge = base["charge"] + effet.get("charge", 0) * progress
                oil_temp = base["oil_temp"] + effet.get("oil_temp", 0) * progress
                voltage = base["voltage"] + effet.get("voltage", 0) * progress
                humidity_adj = humidity_climate + effet.get("humidity", 0) * progress
                ambient_adj = sim_temp + effet.get("ambient", 0) * progress
            else:
                progress = 0.0
                charge, oil_temp, voltage = base["charge"], base["oil_temp"], base["voltage"]
                humidity_adj, ambient_adj = humidity_climate, sim_temp

            features = {"charge": charge, "oil_temp": oil_temp, "ambient": ambient_adj,
                        "humidity": humidity_adj, "voltage": voltage, "season_label": "Saison sèche chaude"}
            risk = predict_risk(features, t["type"])
            push_history(t["id"], risk, features)
            slope, trend_cat = compute_trend(t["id"])
            attn = attention_score(risk, slope)
            band_label, emoji = risk_band(attn)
            log_event_if_needed(t["id"], t["nom"], band_label, risk, active_failure)

            st.metric("Risque courant", f"{risk:.1f} %", delta=f"{slope:+.2f} pts/min")
            st.caption(f"🌡️ T° huile estimée : {oil_temp:.1f} °C · 💧 Humidité perçue : {humidity_adj:.0f} %"
                       + (" · ⚡ orage actif" if weather["orage"] else ""))

            if active_failure:
                st.progress(min(1.0, progress), text=f"Progression de la panne : {progress*100:.0f} %")
                phase_label, phase_kind, phase_text = failure_phase(progress)
                getattr(st, phase_kind)(f"{phase_label} — {phase_text}")
                st.markdown(f"**Pronostic :** {prognosis_text(risk, slope, active_failure)}")
                st.markdown("**Actions réseau recommandées :**")
                for action in NETWORK_ACTIONS_BY_FAILURE.get(active_failure, []):
                    st.markdown(f"- {action}")
                st.markdown(f"- {NETWORK_ACTIONS_BY_TYPE.get(t['type'], '')}")
                with st.expander("🩺 Diagnostic et directives correctives ciblées"):
                    for d in FAILURE_MODES[active_failure]["directives"]:
                        st.markdown(f"- {d}")
            else:
                st.success(f"{emoji} **Fonctionnement normal** ({trend_cat})")

            hist = list(st.session_state.history[t["id"]])
            if len(hist) >= 2:
                dfh = pd.DataFrame(hist)
                fig = px.line(dfh, x="t", y="risk", title="Historique du risque")
                fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"simhist_{t['id']}")

# ============================================================================
# SECTION — ÉTAT DES TRANSFORMATEURS
# ============================================================================
elif section == "📋 État des transformateurs":
    st.subheader("État courant du parc")
    rows = []
    for t in TRANSFORMERS:
        hist = st.session_state.history[t["id"]]
        last = hist[-1] if hist else None
        slope, trend_cat = compute_trend(t["id"])
        risk = last["risk"] if last else 0.0
        attn = attention_score(risk, slope)
        band_label, emoji = risk_band(attn)
        active_failure, progress = get_failure_progress(t["id"], t["type"])
        rows.append({
            "N°": t["id"], "Nom": t["nom"], "Type": t["type"], "Quartier": t["quartier"],
            "Charge": f"{last['charge']:.2f}" if last else "—",
            "T° huile (°C)": f"{last['oil_temp']:.1f}" if last else "—",
            "Tension (p.u.)": f"{last['voltage']:.2f}" if last else "—",
            "Risque (%)": round(risk, 1), "Tendance": trend_cat, "Statut": f"{emoji} {band_label}",
            "Panne simulée": active_failure or "Aucune",
            "Progression panne (%)": round(progress * 100, 1) if active_failure else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Ce tableau reflète l'état le plus récent calculé dans les sections Mode manuel et Simulation, "
               "y compris la panne actuellement simulée sur chaque poste et sa progression, s'il y en a une en cours.")

# ============================================================================
# SECTION — HISTORIQUE DES PANNES
# ============================================================================
elif section == "📈 Historique des pannes":
    st.subheader("Historique des épisodes à risque élevé")
    if not st.session_state.event_log:
        st.info("Aucun épisode à risque élevé enregistré pour l'instant.")
    else:
        rows = []
        for ev in reversed(st.session_state.event_log):
            fin = ev["fin"]
            duree = ((fin or datetime.now()) - ev["debut"]).total_seconds() / 60
            rows.append({
                "Transformateur": ev["nom"], "Cause": ev["type_panne"],
                "Début": ev["debut"].strftime("%H:%M:%S"),
                "Fin": fin.strftime("%H:%M:%S") if fin else "en cours",
                "Durée (min)": round(duree, 1), "Risque max (%)": round(ev["risque_max"], 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Un épisode est ouvert dès qu'un transformateur atteint le niveau de risque « Élevé », "
               "et clôturé lorsqu'il en ressort. Cet historique alimente le calcul de tendance du modèle.")

# ============================================================================
# SECTION — MÉTÉO & PRÉVISIONS
# ============================================================================
elif section == "🌦️ Météo & prévisions":
    st.subheader("Météo en direct et prévisions à 7 jours")
    source = "en direct (Open-Meteo)" if live_ok else "estimation saisonnière (hors ligne)"
    c1, c2 = st.columns(2)
    c1.metric(f"Température actuelle — {source}", f"{ambient_live:.1f} °C")
    c2.metric("Humidité relative actuelle", f"{humidity_live:.0f} %")

    rows, forecast_ok = get_weekly_forecast()
    st.caption("Prévisions " + ("en direct (Open-Meteo)." if forecast_ok else
               "estimées hors ligne (à titre indicatif) — connexion indisponible."))
    df = pd.DataFrame(rows)
    df["Jour"] = pd.to_datetime(df["date"]).dt.strftime("%a %d/%m")
    st.dataframe(df[["Jour", "tmax", "tmin", "pluie_pct", "chaleur_pct"]].rename(columns={
        "tmax": "T° max (°C)", "tmin": "T° min (°C)",
        "pluie_pct": "Probabilité de pluie (%)", "chaleur_pct": "Indice de forte chaleur (%)",
    }), use_container_width=True, hide_index=True)

    fig = px.bar(df, x="Jour", y=["pluie_pct", "chaleur_pct"], barmode="group",
                 labels={"value": "%", "variable": "Indicateur"},
                 title="Probabilité de pluie et indice de forte chaleur — 7 prochains jours")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"L'indice de forte chaleur est une estimation interne fondée sur l'écart entre la "
               f"température maximale prévue et le seuil de vigilance retenu ({HEAT_ALERT_THRESHOLD:.0f} °C) ; "
               "la probabilité de pluie provient directement des prévisions météorologiques. Pour tester "
               "l'impact du climat sur un transformateur précis, utilisez les curseurs météo de la section "
               "« Simulation de pannes ».")

# ============================================================================
# SECTION — CARTE DU PARC
# ============================================================================
elif section == "🗺️ Carte du parc":
    st.subheader("Localisation, risque et obstacles d'intervention")
    rows = []
    for t in TRANSFORMERS:
        hist = st.session_state.history[t["id"]]
        last_risk = hist[-1]["risk"] if hist else 0.0
        slope, trend_cat = compute_trend(t["id"])
        attn = attention_score(last_risk, slope)
        band_label, emoji = risk_band(attn)
        rows.append({"N°": t["id"], "Nom": t["nom"], "Type": t["type"], "Quartier": t["quartier"],
                     "Risque (%)": round(last_risk, 1), "Statut": f"{emoji} {band_label}",
                     "lat": t["lat"], "lon": t["lon"], "attn": attn})
    df_parc = pd.DataFrame(rows)
    st.dataframe(df_parc.drop(columns=["lat", "lon", "attn"]), use_container_width=True, hide_index=True)

    def color_for(attn):
        # Rouge = Élevé, Jaune = Modéré / Vigilance avancée, Vert = Faible — mêmes seuils que RISK_BANDS.
        if attn >= 65:
            return [220, 40, 40, 230]
        if attn >= 35:
            return [240, 200, 30, 230]
        return [40, 160, 70, 230]

    df_parc["color"] = df_parc["attn"].apply(color_for)
    # Petit point fixe posé sur l'emplacement du poste (et non une zone qui grossit avec le
    # risque) : seule sa couleur change, pour un rendu plus proche d'un repère sur une carte réelle.
    layer = pdk.Layer(
        "ScatterplotLayer", data=df_parc, get_position="[lon, lat]",
        get_fill_color="color", get_radius=35, radius_min_pixels=6, radius_max_pixels=14,
        stroked=True, get_line_color=[30, 30, 30, 220], line_width_min_pixels=1, pickable=True,
    )
    view_state = pdk.ViewState(latitude=OUAGA_LAT, longitude=OUAGA_LON, zoom=13.5, pitch=0)
    tooltip = {"text": "{Nom} ({Type}) — {Quartier}\nRisque : {Risque (%)} % — {Statut}"}
    # Fond de carte routier (rues, quartiers, repères), plus lisible et réaliste qu'un simple
    # fond clair : ne nécessite pas de jeton Mapbox (rendu via les tuiles Carto de pydeck).
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        map_style="road",
    ))

    st.markdown("**Obstacles et contraintes d'accès relevés autour de chaque poste :**")
    for t in TRANSFORMERS:
        with st.expander(f"{t['nom']} — {t['quartier']} ({t['type']})"):
            for o in t["obstacles"]:
                st.markdown(f"- {o}")

st.caption(
    "⚠️ Application de démonstration : les grandeurs électriques du mode manuel, les pannes du "
    "mode simulation et la météo simulée sont générées par l'application ; la météo affichée en "
    "en-tête et dans l'onglet dédié est récupérée en direct lorsque la connexion le permet, avec "
    "un repli saisonnier sinon."
)

# ============================================================================
# BOUCLE D'ACTUALISATION — fait avancer l'horloge et les simulations chaque seconde
# ============================================================================
if st.session_state.running:
    time.sleep(1)
    st.rerun()
