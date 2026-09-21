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

st.set_page_config(
    page_title="IA Réseau Pro — Parc SONABEL",
    page_icon="⚡",
    layout="wide",
)

MODEL_PATH = "modele_xgboost.pkl"

HISTORY_LEN = 120
ANIMATION_STEP = 0.18

OUAGA_LAT = 12.3714
OUAGA_LON = -1.5197

FAILURE_TIME_CONSTANT = 90.0
HEAT_ALERT_THRESHOLD = 38.0

# ============================================================================
# TRANSFORMATEURS
# ============================================================================

TRANSFORMERS = [
    {
        "id": 1,
        "nom": "TR-01",
        "type": "Cabine maçonnée",
        "quartier": "Ouaga 2000",
        "lat": 12.3350,
        "lon": -1.4810,
        "obstacles": [
            "Immeubles administratifs proches (accès camion facile)",
            "Voie bitumée dégagée jusqu'au poste",
            "Peu de végétation à proximité immédiate",
        ],
        "vulnerabilite": (
            "Faible exposition climatique directe ; bonne protection "
            "mécanique ; accès aisé pour la maintenance."
        ),
    },
    {
        "id": 2,
        "nom": "TR-02",
        "type": "Préfabriqué",
        "quartier": "Gounghin",
        "lat": 12.3721,
        "lon": -1.5310,
        "obstacles": [
            "Marché de Gounghin à proximité (forte affluence, stationnement anarchique)",
            "Ligne aérienne basse tension croisant l'accès",
            "Caniveau d'évacuation des eaux pluviales longeant le poste",
        ],
        "vulnerabilite": (
            "Exposition modérée ; accès parfois gêné par l'activité "
            "commerciale environnante."
        ),
    },
    {
        "id": 3,
        "nom": "TR-03",
        "type": "Haut de poteau",
        "quartier": "Tanghin",
        "lat": 12.4010,
        "lon": -1.4870,
        "obstacles": [
            "Grand arbre à moins de 5 m (risque de chute de branches sur la ligne)",
            "Proximité du barrage de Tanghin (humidité ambiante élevée, berges)",
            "Habitations rapprochées limitant la manœuvre d'une nacelle",
        ],
        "vulnerabilite": (
            "Forte exposition climatique (foudre, vent, humidité du barrage) ; "
            "accès à la nacelle parfois difficile."
        ),
    },
]

# ============================================================================
# MODES DE PANNE
# ============================================================================

FAILURE_MODES = {
    "Surtension (foudre / manœuvre)": {
        "effet": {
            "voltage": +0.22,
            "oil_temp": +6,
            "charge": +0.05,
        },
        "directives": [
            "Vérifier l'état des parafoudres et des mises à la terre du poste.",
            "Contrôler l'absence de traces d'amorçage sur les traversées.",
            "Programmer un essai de rigidité diélectrique de l'huile dans les 48 h.",
        ],
    },

    "Court-circuit interne": {
        "effet": {
            "charge": +0.35,
            "oil_temp": +18,
            "voltage": -0.15,
        },
        "directives": [
            "Mettre hors tension immédiatement et consigner le transformateur.",
            "Réaliser une analyse des gaz dissous (DGA) avant toute remise en service.",
            "Inspecter le serrage des enroulements et l'état des connexions internes.",
        ],
    },

    "Température élevée / surcharge thermique": {
        "effet": {
            "oil_temp": +14,
            "ambient": +4,
            "charge": +0.15,
        },
        "directives": [
            "Réduire la charge sur ce transformateur en reportant une partie des abonnés voisins.",
            "Nettoyer les radiateurs et vérifier l'absence d'encrassement par la poussière.",
            "Surveiller le thermomètre à cadran toutes les heures jusqu'à stabilisation.",
        ],
    },

    "Surcharge électrique prolongée": {
        "effet": {
            "charge": +0.40,
            "oil_temp": +10,
        },
        "directives": [
            "Vérifier le taux de charge réel par rapport à la puissance nominale.",
            "Planifier un rééquilibrage des phases si un déséquilibre est constaté.",
            "Étudier un changement de transformateur pour une puissance supérieure si la surcharge est durable.",
        ],
    },

    "Défaut d'isolement (humidité)": {
        "effet": {
            "humidity": +25,
            "oil_temp": +5,
            "voltage": -0.08,
        },
        "directives": [
            "Contrôler l'étanchéité du conservateur d'huile et l'état de l'assécheur d'air.",
            "Mesurer la teneur en eau de l'huile (méthode Karl Fischer) et la résistance d'isolement.",
            "Vérifier l'absence d'infiltration d'eau dans les boîtes de jonction basse tension.",
        ],
    },
}

# ============================================================================
# ACTIONS RÉSEAU
# ============================================================================

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
        "Limiter temporairement les nouveaux branchements sur ce départ.",
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

NETWORK_ACTIONS_BY_TYPE = {
    "Cabine maçonnée": (
        "Accès véhicule dégagé : une équipe peut intervenir rapidement, "
        "sans contrainte majeure."
    ),

    "Préfabriqué": (
        "Prévenir les commerçants et usagers à proximité avant toute manœuvre : "
        "l'accès peut être encombré."
    ),

    "Haut de poteau": (
        "Intervention en hauteur : mobiliser une nacelle et sécuriser le "
        "dégagement autour du poteau avant toute coupure."
    ),
}

# ============================================================================
# MÉTÉO
# ============================================================================

WEATHER_PRESETS = {
    "☀️ Ciel dégagé (normal)": {
        "humidity_bonus": 0,
        "orage": False,
    },

    "🌧️ Pluie": {
        "humidity_bonus": 30,
        "orage": False,
    },

    "⛈️ Orage (pluie + foudre)": {
        "humidity_bonus": 35,
        "orage": True,
    },

    "🌬️ Fraîcheur / harmattan frais": {
        "humidity_bonus": -15,
        "orage": False,
    },

    "🔥 Canicule": {
        "humidity_bonus": -10,
        "orage": False,
    },
}

TYPE_TIME_FACTOR = {
    "Cabine maçonnée": 1.35,
    "Préfabriqué": 1.0,
    "Haut de poteau": 0.7,
}

TYPE_CLIMATE = {
    "Cabine maçonnée": {
        "ambient_factor": 0.5,
        "humidity_factor": 0.35,
        "orage_voltage_bonus": 0.02,
    },

    "Préfabriqué": {
        "ambient_factor": 0.85,
        "humidity_factor": 0.7,
        "orage_voltage_bonus": 0.05,
    },

    "Haut de poteau": {
        "ambient_factor": 1.2,
        "humidity_factor": 1.0,
        "orage_voltage_bonus": 0.10,
    },
}

# ============================================================================
# SEUILS DE RISQUE
# ============================================================================

RISK_BANDS = [
    (0, 35, "Faible", "🟢"),
    (35, 50, "Modéré", "🟠"),
    (50, 65, "Vigilance avancée", "🟡"),
    (65, 101, "Élevé", "🔴"),
]

SECTIONS = [
    "🖐️ Mode manuel",
    "🎲 Simulation de pannes",
    "📋 État des transformateurs",
    "📈 Historique des pannes",
    "🌦️ Météo & prévisions",
    "🗺️ Carte du parc",
]

# ============================================================================
# FONCTIONS RISQUE
# ============================================================================

def risk_band(pct: float):
    pct = float(np.clip(pct, 0, 100))

    for lo, hi, label, emoji in RISK_BANDS:
        if lo <= pct < hi:
            return label, emoji

    return "Élevé", "🔴"

# ============================================================================
# CHARGEMENT DU MODÈLE
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

TYPE_MULTIPLIER = {
    "Cabine maçonnée": 0.85,
    "Préfabriqué": 1.0,
    "Haut de poteau": 1.20,
}

def predict_risk(features: dict, ttype: str) -> float:

    season_map = {
        "Saison sèche fraîche": 0,
        "Saison sèche chaude": 1,
        "Harmattan": 2,
        "Saison des pluies": 3,
    }

    season_code = season_map.get(
        features.get("season_label", "Saison sèche chaude"),
        1,
    )

    if MODEL is not None:
        try:
            x = pd.DataFrame([
                {
                    "charge": features["charge"],
                    "oil": features["oil_temp"],
                    "ambient": features["ambient"],
                    "humidity": features["humidity"],
                    "voltage": features["voltage"],
                    "season": season_code,
                }
            ])

            base = float(
                MODEL.predict_proba(x)[0][1] * 100
            )

        except Exception:
            base = _heuristic_risk(features, season_code)

    else:
        base = _heuristic_risk(features, season_code)

    return float(
        np.clip(
            base * TYPE_MULTIPLIER.get(ttype, 1.0),
            0,
            100,
        )
    )

def _heuristic_risk(f: dict, season_code: int):

    charge_term = max(0.0, f["charge"] - 0.8) * 55

    oil_term = max(
        0.0,
        f["oil_temp"] - 65,
    ) * 1.6

    ambient_term = max(
        0.0,
        f["ambient"] - 32,
    ) * 1.2

    humidity_term = max(
        0.0,
        f["humidity"] - 60,
    ) * 0.5

    voltage_term = abs(
        f["voltage"] - 1.0
    ) * 60

    season_term = {
        0: 0,
        1: 8,
        2: 5,
        3: 10,
    }.get(season_code, 0)

    total = (
        8
        + charge_term
        + oil_term
        + ambient_term
        + humidity_term
        + voltage_term
        + season_term
    )

    return float(
        np.clip(total, 0, 100)
    )

# ============================================================================
# ÉTAT DE SESSION
# ============================================================================

def create_last_status():

    return {
        t["id"]: {
            "risk": 0.0,
            "attention": 0.0,
            "trend": 0.0,
            "trend_cat": "stable",
            "status": "Faible",
            "emoji": "🟢",
            "source": "initial",
            "failure": None,
            "progress": 0.0,
            "features": {},
        }
        for t in TRANSFORMERS
    }

def init_state():

    defaults = {
        "running": True,

        "section": SECTIONS[0],

        "history": lambda: {
            t["id"]: deque(maxlen=HISTORY_LEN)
            for t in TRANSFORMERS
        },

        "manual_target": lambda: {
            t["id"]: {
                "voltage": 1.0,
                "charge": 0.8,
                "current_ratio": 0.8,
            }
            for t in TRANSFORMERS
        },

        "manual_effective": lambda: {
            t["id"]: {
                "voltage": 1.0,
                "charge": 0.8,
                "current_ratio": 0.8,
            }
            for t in TRANSFORMERS
        },

        "sim_failure": lambda: {
            t["id"]: None
            for t in TRANSFORMERS
        },

        "sim_start_time": lambda: {
            t["id"]: None
            for t in TRANSFORMERS
        },

        "event_log": list,

        "active_event": lambda: {
            t["id"]: None
            for t in TRANSFORMERS
        },

        # État persistant utilisé notamment par la carte
        "last_status": create_last_status,
    }

    for key, default in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = (
                default()
                if callable(default)
                else default
            )

    # Sécurité pour les anciens états de session

    for t in TRANSFORMERS:

        st.session_state.manual_target.setdefault(
            t["id"],
            {
                "voltage": 1.0,
                "charge": 0.8,
                "current_ratio": 0.8,
            },
        )

        st.session_state.manual_effective.setdefault(
            t["id"],
            {
                "voltage": 1.0,
                "charge": 0.8,
                "current_ratio": 0.8,
            },
        )

        st.session_state.sim_failure.setdefault(
            t["id"],
            None,
        )

        st.session_state.sim_start_time.setdefault(
            t["id"],
            None,
        )

        st.session_state.active_event.setdefault(
            t["id"],
            None,
        )

        if t["id"] not in st.session_state.history:

            st.session_state.history[
                t["id"]
            ] = deque(maxlen=HISTORY_LEN)

    if "last_status" not in st.session_state:

        st.session_state.last_status = create_last_status()

    for t in TRANSFORMERS:

        st.session_state.last_status.setdefault(
            t["id"],
            {
                "risk": 0.0,
                "attention": 0.0,
                "trend": 0.0,
                "trend_cat": "stable",
                "status": "Faible",
                "emoji": "🟢",
                "source": "initial",
                "failure": None,
                "progress": 0.0,
                "features": {},
            },
        )

# ============================================================================
# HISTORIQUE
# ============================================================================

def push_history(t_id, risk_pct, features):

    st.session_state.history[t_id].append(
        {
            "t": datetime.now(),
            "risk": risk_pct,
            **features,
        }
    )

def compute_trend(t_id):

    hist = st.session_state.history[t_id]

    if len(hist) < 3:
        return 0.0, "stable"

    recent = list(hist)[-12:]

    t0 = recent[0]["t"]

    xs = [
        (p["t"] - t0).total_seconds() / 60.0
        for p in recent
    ]

    ys = [
        p["risk"]
        for p in recent
    ]

    if xs[-1] - xs[0] < 1e-6:
        return 0.0, "stable"

    slope = np.polyfit(
        xs,
        ys,
        1,
    )[0]

    if slope < 0.5:
        cat = "stable"

    elif slope < 2.5:
        cat = "en hausse"

    elif slope < 6:
        cat = "en hausse rapide"

    else:
        cat = "critique — hausse brutale"

    return float(slope), cat

def update_last_status(
    t_id,
    risk,
    slope,
    trend_cat,
    band_label,
    emoji,
    source,
    failure=None,
    progress=0.0,
    features=None,
):

    st.session_state.last_status[t_id] = {
        "risk": float(risk),
        "attention": float(risk),
        "trend": float(slope),
        "trend_cat": trend_cat,
        "status": band_label,
        "emoji": emoji,
        "source": source,
        "failure": failure,
        "progress": float(progress),
        "features": features or {},
    }

# ============================================================================
# MANUEL
# ============================================================================

def animate_towards_target(t_id):

    tgt = st.session_state.manual_target[t_id]

    eff = st.session_state.manual_effective[t_id]

    for k in tgt:

        eff[k] += (
            tgt[k] - eff[k]
        ) * ANIMATION_STEP

        if abs(
            eff[k] - tgt[k]
        ) < 1e-3:

            eff[k] = tgt[k]

# ============================================================================
# ÉVÉNEMENTS
# ============================================================================

def log_event_if_needed(
    t_id,
    nom,
    band_label,
    risk,
    panne_label,
):

    active = st.session_state.active_event[t_id]

    if band_label == "Élevé":

        if active is None:

            st.session_state.event_log.append(
                {
                    "transfo_id": t_id,
                    "nom": nom,
                    "type_panne": (
                        panne_label
                        or "Mode manuel"
                    ),
                    "debut": datetime.now(),
                    "fin": None,
                    "risque_max": risk,
                }
            )

            st.session_state.active_event[t_id] = (
                len(
                    st.session_state.event_log
                ) - 1
            )

        else:

            ev = st.session_state.event_log[
                active
            ]

            ev["risque_max"] = max(
                ev["risque_max"],
                risk,
            )

    else:

        if active is not None:

            st.session_state.event_log[
                active
            ]["fin"] = datetime.now()

            st.session_state.active_event[
                t_id
            ] = None

# ============================================================================
# PROGRESSION PANNE
# ============================================================================

def get_failure_progress(
    t_id,
    ttype,
):

    active = (
        st.session_state
        .sim_failure
        .get(t_id)
    )

    if not active:
        return None, 0.0

    start = (
        st.session_state
        .sim_start_time
        .get(t_id)
    )

    if start is None:
        return active, 0.0

    elapsed = max(
        0.0,
        time.time() - start,
    )

    effective_time_constant = (
        FAILURE_TIME_CONSTANT
        * TYPE_TIME_FACTOR.get(
            ttype,
            1.0,
        )
    )

    progress = (
        1
        - math.exp(
            -elapsed
            / effective_time_constant
        )
    )

    return active, progress

# ============================================================================
# PHASE DE PANNE
# ============================================================================

def failure_phase(progress):

    if progress < 0.12:

        return (
            "🔍 Anomalie naissante",
            "info",
            (
                "Un écart encore ténu vient d'apparaître "
                "sur ce transformateur."
            ),
        )

    if progress < 0.45:

        return (
            "⚠️ Dégradation en cours",
            "warning",
            (
                "L'écart se creuse progressivement. "
                "Une intervention préventive peut limiter "
                "l'aggravation."
            ),
        )

    if progress < 0.8:

        return (
            "🟠 Aggravation avancée",
            "warning",
            (
                "La dégradation s'accélère. "
                "L'état critique approche."
            ),
        )

    return (
        "🚨 État critique",
        "error",
        (
            "Le transformateur approche d'un état "
            "de défaillance critique."
        ),
    )

# ============================================================================
# PRONOSTIC
# ============================================================================

def prognosis_text(
    risk_pct,
    slope,
    failure_label=None,
):

    label = (
        failure_label
        or "défaillance"
    )

    if slope <= 0.15:

        return (
            f"Évolution stable — pas d'aggravation "
            f"significative détectée pour "
            f"{label.lower()}."
        )

    remaining_pct = max(
        0.0,
        90 - risk_pct,
    )

    minutes_to_90 = (
        remaining_pct / slope
        if slope > 0
        else None
    )

    if (
        minutes_to_90 is None
        or minutes_to_90 > 24 * 60
    ):

        return (
            f"Tendance à la hausse lente pour "
            f"{label.lower()} — surveillance renforcée."
        )

    hours = minutes_to_90 / 60

    delay = (
        f"environ {int(minutes_to_90)} min"
        if hours < 1
        else f"environ {hours:.1f} h"
    )

    return (
        f"Risque de {label.lower()} à venir : "
        f"{min(90, risk_pct + slope * 30):.0f} % "
        f"dans {delay} si rien n'est fait."
    )

# ============================================================================
# MÉTÉO LIVE
# ============================================================================

@st.cache_data(ttl=600)
def get_live_weather():

    try:

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={OUAGA_LAT}"
            f"&longitude={OUAGA_LON}"
            f"&current=temperature_2m,"
            f"relative_humidity_2m"
            f"&timezone=Africa%2FOuagadougou"
        )

        data = requests.get(
            url,
            timeout=4,
        ).json()

        return (
            float(
                data["current"][
                    "temperature_2m"
                ]
            ),
            float(
                data["current"][
                    "relative_humidity_2m"
                ]
            ),
            True,
        )

    except Exception:

        month = datetime.now().month

        temp = (
            38
            if month in (3, 4, 5)
            else (
                28
                if month in (7, 8, 9)
                else 33
            )
        )

        hum = (
            75
            if month in (7, 8, 9)
            else 25
        )

        return (
            float(temp),
            float(hum),
            False,
        )

# ============================================================================
# PRÉVISIONS 7 JOURS
# ============================================================================

@st.cache_data(ttl=1800)
def get_weekly_forecast():

    try:

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={OUAGA_LAT}"
            f"&longitude={OUAGA_LON}"
            f"&daily=temperature_2m_max,"
            f"temperature_2m_min,"
            f"precipitation_probability_max"
            f"&forecast_days=7"
            f"&timezone=Africa%2FOuagadougou"
        )

        data = requests.get(
            url,
            timeout=5,
        ).json()

        d = data["daily"]

        rows = []

        for i, date in enumerate(
            d["time"]
        ):

            tmax = float(
                d["temperature_2m_max"][i]
            )

            tmin = float(
                d["temperature_2m_min"][i]
            )

            rain_pct = float(
                d[
                    "precipitation_probability_max"
                ][i]
            )

            heat_pct = float(
                np.clip(
                    (
                        tmax - 32
                    )
                    / 13
                    * 100,
                    0,
                    100,
                )
            )

            rows.append(
                {
                    "date": date,
                    "tmax": tmax,
                    "tmin": tmin,
                    "pluie_pct": rain_pct,
                    "chaleur_pct": heat_pct,
                }
            )

        return rows, True

    except Exception:

        month = datetime.now().month

        rainy = month in (
            6,
            7,
            8,
            9,
            10,
        )

        rows = []

        for i in range(7):

            d = (
                datetime.now()
                + timedelta(days=i)
            )

            tmax = (
                36
                + np.random.uniform(-3, 4)
                if not rainy
                else 31
                + np.random.uniform(-2, 3)
            )

            rain_pct = (
                np.random.uniform(40, 85)
                if rainy
                else np.random.uniform(0, 15)
            )

            heat_pct = float(
                np.clip(
                    (tmax - 32)
                    / 13
                    * 100,
                    0,
                    100,
                )
            )

            rows.append(
                {
                    "date": d.strftime(
                        "%Y-%m-%d"
                    ),
                    "tmax": round(
                        tmax,
                        1,
                    ),
                    "tmin": round(
                        tmax - 8,
                        1,
                    ),
                    "pluie_pct": round(
                        rain_pct,
                        0,
                    ),
                    "chaleur_pct": round(
                        heat_pct,
                        0,
                    ),
                }
            )

        return rows, False

# ============================================================================
# INITIALISATION
# ============================================================================

init_state()

# ============================================================================
# EN-TÊTE
# ============================================================================

st.title(
    "⚡ Système IA de Prédiction — "
    "Parc de transformateurs (Ouagadougou)"
)

ambient_live, humidity_live, live_ok = (
    get_live_weather()
)

top1, top2, top3 = st.columns(
    [2, 2, 3]
)

with top1:

    if st.button(
        "⏸️ Pause"
        if st.session_state.running
        else "▶️ Reprendre",
        key="run_toggle",
    ):

        st.session_state.running = (
            not st.session_state.running
        )

with top2:

    st.metric(
        "🕒 Heure actuelle",
        datetime.now().strftime(
            "%H:%M:%S"
        ),
    )

    st.caption(
        datetime.now().strftime(
            "%A %d %B %Y"
        )
    )

with top3:

    source = (
        "en direct (Open-Meteo)"
        if live_ok
        else "estimation saisonnière (hors ligne)"
    )

    st.metric(
        f"🌡️ Météo Ouagadougou — {source}",
        f"{ambient_live:.1f} °C / "
        f"{humidity_live:.0f} % HR",
    )

st.caption(
    "L'horloge affiche l'heure réelle. "
    "Tant que la simulation n'est pas en pause, "
    "la page se rafraîchit chaque seconde."
)

# ============================================================================
# COMPARAISON DES TRANSFORMATEURS
# ============================================================================

with st.expander(
    "📋 Étude comparative des 3 types de transformateurs du parc"
):

    st.table(
        pd.DataFrame(
            [
                {
                    "Type": t["type"],
                    "Quartier": t["quartier"],
                    "Vulnérabilité type": t[
                        "vulnerabilite"
                    ],
                }
                for t in TRANSFORMERS
            ]
        )
    )

st.divider()

# ============================================================================
# NAVIGATION
# ============================================================================

nav_cols = st.columns(
    len(SECTIONS)
)

for col, sec in zip(
    nav_cols,
    SECTIONS,
):

    is_active = (
        st.session_state.section
        == sec
    )

    if col.button(
        sec,
        key=f"nav_{sec}",
        use_container_width=True,
        type=(
            "primary"
            if is_active
            else "secondary"
        ),
    ):

        st.session_state.section = sec

st.divider()

section = st.session_state.section

# ============================================================================
# SECTION 1 — MODE MANUEL
# ============================================================================

if section == "🖐️ Mode manuel":

    st.subheader(
        "Mode manuel — paramètres électriques"
    )

    st.caption(
        "Ce mode permet de modifier indépendamment "
        "les grandeurs électriques des transformateurs. "
        "Les pannes simulées sont disponibles uniquement "
        "dans la section « Simulation de pannes »."
    )

    cols = st.columns(3)

    for col, t in zip(
        cols,
        TRANSFORMERS,
    ):

        with col:

            st.markdown(
                f"### {t['nom']} — {t['type']}"
            )

            st.caption(
                f"📍 {t['quartier']}"
            )

            tgt = (
                st.session_state
                .manual_target[t["id"]]
            )

            tgt["voltage"] = st.slider(
                "Tension (p.u.)",
                0.85,
                1.15,
                tgt["voltage"],
                0.01,
                key=f"v_{t['id']}",
            )

            tgt["current_ratio"] = st.slider(
                "Courant relatif",
                0.2,
                1.6,
                tgt["current_ratio"],
                0.02,
                key=f"c_{t['id']}",
            )

            tgt["charge"] = st.slider(
                "Charge relative",
                0.2,
                1.6,
                tgt["charge"],
                0.02,
                key=f"ch_{t['id']}",
            )

            animate_towards_target(
                t["id"]
            )

            eff = (
                st.session_state
                .manual_effective[t["id"]]
            )

            oil_temp = (
                45
                + eff["charge"] * 28
                + max(
                    0,
                    ambient_live - 30,
                )
                * 0.6
            )

            features = {
                "charge": eff["charge"],
                "oil_temp": oil_temp,
                "ambient": ambient_live,
                "humidity": humidity_live,
                "voltage": eff["voltage"],
                "season_label": (
                    "Saison sèche chaude"
                ),
            }

            risk = predict_risk(
                features,
                t["type"],
            )

            push_history(
                t["id"],
                risk,
                features,
            )

            slope, trend_cat = (
                compute_trend(
                    t["id"]
                )
            )

            band_label, emoji = risk_band(
                risk
            )

            update_last_status(
                t["id"],
                risk,
                slope,
                trend_cat,
                band_label,
                emoji,
                source="manuel",
                failure=None,
                progress=0.0,
                features=features,
            )

            st.metric(
                "Risque électrique",
                f"{risk:.1f} %",
                delta=f"{slope:+.2f} pts/min",
            )

            if risk >= 65:

                st.error(
                    f"🚨 **Risque élevé — "
                    f"{risk:.1f} %**"
                )

            elif risk >= 50:

                st.warning(
                    f"⚠️ **INTERPELLATION — "
                    f"{risk:.1f} %**"
                )

            elif risk >= 35:

                st.warning(
                    f"🟠 **Risque modéré — "
                    f"{risk:.1f} %**"
                )

            else:

                st.success(
                    f"🟢 **Risque faible — "
                    f"{risk:.1f} %**"
                )

# ============================================================================
# SECTION 2 — SIMULATION DE PANNES
# ============================================================================

elif section == "🎲 Simulation de pannes":

    st.subheader(
        "Simulation — évolution progressive vers la panne"
    )

    st.caption(
        "Lorsqu'une panne est sélectionnée, "
        "le risque commence près de 0 % puis augmente "
        "progressivement. À 50 %, une interpellation "
        "est déclenchée. À 65 %, le niveau devient élevé."
    )

    # ------------------------------------------------------------------------
    # MÉTÉO SIMULÉE
    # ------------------------------------------------------------------------

    st.markdown(
        "#### 🌦️ Conditions climatiques simulées"
    )

    wc1, wc2 = st.columns(2)

    with wc1:

        sim_temp = st.slider(
            "Température ambiante simulée (°C)",
            15,
            50,
            int(
                round(
                    np.clip(
                        ambient_live,
                        15,
                        50,
                    )
                )
            ),
            1,
            key="sim_temp_slider",
        )

    with wc2:

        sim_weather_label = st.selectbox(
            "Conditions météo simulées",
            list(
                WEATHER_PRESETS.keys()
            ),
            key="sim_weather_select",
        )

    weather = WEATHER_PRESETS[
        sim_weather_label
    ]

    st.caption(
        "La température et les conditions choisies "
        "influencent la simulation selon le type de "
        "transformateur."
    )

    # ------------------------------------------------------------------------
    # TRANSFORMATEURS
    # ------------------------------------------------------------------------

    cols = st.columns(3)

    for col, t in zip(
        cols,
        TRANSFORMERS,
    ):

        with col:

            st.markdown(
                f"### {t['nom']} — {t['type']}"
            )

            st.caption(
                f"📍 {t['quartier']}"
            )

            options = [
                "Aucune (fonctionnement normal)"
            ] + list(
                FAILURE_MODES.keys()
            )

            current_choice = (
                st.session_state
                .sim_failure
                .get(t["id"])
                or options[0]
            )

            choice = st.selectbox(
                "Mode de panne simulé",
                options,
                index=(
                    options.index(
                        current_choice
                    )
                    if current_choice in options
                    else 0
                ),
                key=f"fail_{t['id']}",
            )

            previous_choice = (
                st.session_state
                .sim_failure
                .get(t["id"])
                or options[0]
            )

            # Nouvelle sélection
            if choice != previous_choice:

                if (
                    choice
                    == options[0]
                ):

                    st.session_state.sim_failure[
                        t["id"]
                    ] = None

                    st.session_state.sim_start_time[
                        t["id"]
                    ] = None

                    # Retour à un état initial
                    # proche de 0 %
                    update_last_status(
                        t["id"],
                        2.0,
                        0.0,
                        "stable",
                        "Faible",
                        "🟢",
                        source="simulation",
                        failure=None,
                        progress=0.0,
                        features={},
                    )

                else:

                    st.session_state.sim_failure[
                        t["id"]
                    ] = choice

                    st.session_state.sim_start_time[
                        t["id"]
                    ] = time.time()

                    # Une nouvelle panne commence
                    # à presque 0 %.
                    update_last_status(
                        t["id"],
                        2.0,
                        0.0,
                        "stable",
                        "Faible",
                        "🟢",
                        source="simulation",
                        failure=choice,
                        progress=0.0,
                        features={},
                    )

            active_failure = (
                st.session_state
                .sim_failure
                .get(t["id"])
            )

            climate = TYPE_CLIMATE.get(
                t["type"],
                {
                    "ambient_factor": 1.0,
                    "humidity_factor": 1.0,
                    "orage_voltage_bonus": 0.0,
                },
            )

            # ----------------------------------------------------------------
            # IMPACT DU CLIMAT
            # ----------------------------------------------------------------

            oil_temp_climate_add = (
                max(
                    0.0,
                    sim_temp - 30,
                )
                * 0.6
                * climate[
                    "ambient_factor"
                ]
            )

            humidity_climate = float(
                np.clip(
                    humidity_live
                    + weather[
                        "humidity_bonus"
                    ]
                    * climate[
                        "humidity_factor"
                    ],
                    5,
                    100,
                )
            )

            voltage_climate_add = (
                climate[
                    "orage_voltage_bonus"
                ]
                if weather["orage"]
                else 0.0
            )

            base = {
                "charge": 0.8,

                "oil_temp": (
                    45
                    + 0.8 * 28
                    + oil_temp_climate_add
                ),

                "voltage": (
                    1.0
                    + voltage_climate_add
                ),
            }

            # ----------------------------------------------------------------
            # PROGRESSION
            # ----------------------------------------------------------------

            if active_failure:

                start = (
                    st.session_state
                    .sim_start_time
                    .get(t["id"])
                )

                if start is None:

                    start = time.time()

                    st.session_state.sim_start_time[
                        t["id"]
                    ] = start

                elapsed = max(
                    0.0,
                    time.time() - start,
                )

                effective_time_constant = (
                    FAILURE_TIME_CONSTANT
                    * TYPE_TIME_FACTOR.get(
                        t["type"],
                        1.0,
                    )
                )

                progress = (
                    1
                    - math.exp(
                        -elapsed
                        / effective_time_constant
                    )
                )

                effet = FAILURE_MODES[
                    active_failure
                ]["effet"]

                charge = (
                    base["charge"]
                    + effet.get(
                        "charge",
                        0,
                    )
                    * progress
                )

                oil_temp = (
                    base["oil_temp"]
                    + effet.get(
                        "oil_temp",
                        0,
                    )
                    * progress
                )

                voltage = (
                    base["voltage"]
                    + effet.get(
                        "voltage",
                        0,
                    )
                    * progress
                )

                humidity_adj = (
                    humidity_climate
                    + effet.get(
                        "humidity",
                        0,
                    )
                    * progress
                )

                ambient_adj = (
                    sim_temp
                    + effet.get(
                        "ambient",
                        0,
                    )
                    * progress
                )

            else:

                progress = 0.0

                charge = base["charge"]

                oil_temp = base[
                    "oil_temp"
                ]

                voltage = base[
                    "voltage"
                ]

                humidity_adj = (
                    humidity_climate
                )

                ambient_adj = sim_temp

            # ----------------------------------------------------------------
            # GRANDEURS
            # ----------------------------------------------------------------

            features = {
                "charge": charge,
                "oil_temp": oil_temp,
                "ambient": ambient_adj,
                "humidity": humidity_adj,
                "voltage": voltage,
                "season_label": (
                    "Saison sèche chaude"
                ),
            }

            # ----------------------------------------------------------------
            # CALCUL DU RISQUE DE SIMULATION
            # ----------------------------------------------------------------

            if active_failure is None:

                # Fonctionnement normal :
                # risque proche de zéro.
                risk = 2.0

                climate_stress = 0.0

                if sim_temp >= 38:

                    climate_stress += min(
                        (
                            sim_temp - 38
                        )
                        * 0.35,
                        3.0,
                    )

                if humidity_adj >= 85:

                    climate_stress += min(
                        (
                            humidity_adj - 85
                        )
                        * 0.05,
                        2.0,
                    )

                risk = float(
                    np.clip(
                        risk
                        + climate_stress,
                        0,
                        10,
                    )
                )

            else:

                # Courbe progressive.
                #
                # Au début :
                # progress = 0 -> risque ≈ 2 %
                #
                # Puis la montée accélère.
                #
                progress_risk = (
                    100
                    * (
                        progress
                        ** 1.55
                    )
                )

                climate_bonus = 0.0

                if sim_temp > 35:

                    climate_bonus += (
                        (
                            sim_temp - 35
                        )
                        * 0.7
                        * climate[
                            "ambient_factor"
                        ]
                    )

                if humidity_adj > 75:

                    climate_bonus += (
                        (
                            humidity_adj - 75
                        )
                        * 0.10
                        * climate[
                            "humidity_factor"
                        ]
                    )

                if weather["orage"]:

                    climate_bonus += (
                        5.0
                        * climate[
                            "ambient_factor"
                        ]
                    )

                risk = float(
                    np.clip(
                        progress_risk
                        + climate_bonus,
                        0,
                        100,
                    )
                )

            # ----------------------------------------------------------------
            # HISTORIQUE
            # ----------------------------------------------------------------

            push_history(
                t["id"],
                risk,
                features,
            )

            slope, trend_cat = (
                compute_trend(
                    t["id"]
                )
            )

            band_label, emoji = (
                risk_band(risk)
            )

            # ----------------------------------------------------------------
            # SAUVEGARDE ÉTAT POUR LA CARTE
            # ----------------------------------------------------------------

            update_last_status(
                t["id"],
                risk,
                slope,
                trend_cat,
                band_label,
                emoji,
                source="simulation",
                failure=active_failure,
                progress=progress,
                features=features,
            )

            log_event_if_needed(
                t["id"],
                t["nom"],
                band_label,
                risk,
                active_failure,
            )

            # ----------------------------------------------------------------
            # AFFICHAGE RISQUE
            # ----------------------------------------------------------------

            st.metric(
                "Risque courant",
                f"{risk:.1f} %",
                delta=f"{slope:+.2f} pts/min",
            )

            # ----------------------------------------------------------------
            # ALERTES
            # ----------------------------------------------------------------

            if risk >= 65:

                st.error(
                    f"🚨 **ALERTE CRITIQUE — "
                    f"{risk:.1f} %** : "
                    f"le niveau de risque est élevé."
                )

            elif risk >= 50:

                st.warning(
                    f"⚠️ **INTERPELLATION — "
                    f"{risk:.1f} %** : "
                    f"le risque vient de franchir "
                    f"le seuil de 50 %."
                )

            elif risk >= 35:

                st.warning(
                    f"🟠 **Vigilance — "
                    f"{risk:.1f} %** : "
                    f"la dérive devient significative."
                )

            else:

                st.success(
                    f"🟢 **Risque faible — "
                    f"{risk:.1f} %**"
                )

            st.caption(
                f"🌡️ T° huile estimée : "
                f"{oil_temp:.1f} °C · "
                f"💧 Humidité perçue : "
                f"{humidity_adj:.0f} %"
                + (
                    " · ⚡ orage actif"
                    if weather["orage"]
                    else ""
                )
            )

            # ----------------------------------------------------------------
            # PROGRESSION DE PANNE
            # ----------------------------------------------------------------

            if active_failure:

                st.progress(
                    min(
                        1.0,
                        progress,
                    ),
                    text=(
                        "Progression de la panne : "
                        f"{progress * 100:.0f} %"
                    ),
                )

                phase_label, phase_kind, phase_text = (
                    failure_phase(
                        progress
                    )
                )

                getattr(
                    st,
                    phase_kind,
                )(
                    f"{phase_label} — "
                    f"{phase_text}"
                )

                st.markdown(
                    "**Pronostic :** "
                    + prognosis_text(
                        risk,
                        slope,
                        active_failure,
                    )
                )

                st.markdown(
                    "**Actions réseau recommandées :**"
                )

                for action in (
                    NETWORK_ACTIONS_BY_FAILURE.get(
                        active_failure,
                        [],
                    )
                ):

                    st.markdown(
                        f"- {action}"
                    )

                st.markdown(
                    "- "
                    + NETWORK_ACTIONS_BY_TYPE.get(
                        t["type"],
                        "",
                    )
                )

                with st.expander(
                    "🩺 Diagnostic et directives correctives ciblées"
                ):

                    for directive in (
                        FAILURE_MODES[
                            active_failure
                        ]["directives"]
                    ):

                        st.markdown(
                            f"- {directive}"
                        )

            else:

                st.success(
                    "🟢 Fonctionnement normal"
                )

            # ----------------------------------------------------------------
            # GRAPHIQUE
            # ----------------------------------------------------------------

            hist = list(
                st.session_state
                .history[t["id"]]
            )

            if len(hist) >= 2:

                dfh = pd.DataFrame(
                    hist
                )

                fig = px.line(
                    dfh,
                    x="t",
                    y="risk",
                    title="Historique du risque",
                )

                # Ligne seuil 50 %
                fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Interpellation 50 %",
                )

                # Ligne seuil 65 %
                fig.add_hline(
                    y=65,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Risque élevé 65 %",
                )

                fig.update_layout(
                    height=200,
                    margin=dict(
                        l=10,
                        r=10,
                        t=40,
                        b=10,
                    ),
                    showlegend=False,
                    yaxis=dict(
                        range=[0, 100]
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"simhist_{t['id']}",
                )

# ============================================================================
# SECTION 3 — ÉTAT DES TRANSFORMATEURS
# ============================================================================

elif section == "📋 État des transformateurs":

    st.subheader(
        "État courant du parc"
    )

    rows = []

    for t in TRANSFORMERS:

        status = (
            st.session_state
            .last_status[t["id"]]
        )

        risk = status["risk"]

        rows.append(
            {
                "N°": t["id"],
                "Nom": t["nom"],
                "Type": t["type"],
                "Quartier": t["quartier"],
                "Risque (%)": round(
                    risk,
                    1,
                ),
                "Statut": (
                    f"{status['emoji']} "
                    f"{status['status']}"
                ),
                "Source": status[
                    "source"
                ],
                "Panne simulée": (
                    status["failure"]
                    or "Aucune"
                ),
                "Progression panne (%)": (
                    round(
                        status[
                            "progress"
                        ]
                        * 100,
                        1,
                    )
                    if status["failure"]
                    else "—"
                ),
                "Tendance": status[
                    "trend_cat"
                ],
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Le tableau utilise le dernier état enregistré "
        "pour chaque transformateur, y compris l'état "
        "issu de la simulation."
    )

# ============================================================================
# SECTION 4 — HISTORIQUE DES PANNES
# ============================================================================

elif section == "📈 Historique des pannes":

    st.subheader(
        "Historique des épisodes à risque élevé"
    )

    if not st.session_state.event_log:

        st.info(
            "Aucun épisode à risque élevé "
            "enregistré pour l'instant."
        )

    else:

        rows = []

        for ev in reversed(
            st.session_state.event_log
        ):

            fin = ev["fin"]

            duree = (
                (
                    fin
                    or datetime.now()
                )
                - ev["debut"]
            ).total_seconds() / 60

            rows.append(
                {
                    "Transformateur": ev[
                        "nom"
                    ],
                    "Cause": ev[
                        "type_panne"
                    ],
                    "Début": ev[
                        "debut"
                    ].strftime(
                        "%H:%M:%S"
                    ),
                    "Fin": (
                        fin.strftime(
                            "%H:%M:%S"
                        )
                        if fin
                        else "en cours"
                    ),
                    "Durée (min)": round(
                        duree,
                        1,
                    ),
                    "Risque max (%)": round(
                        ev[
                            "risque_max"
                        ],
                        1,
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================================
# SECTION 5 — MÉTÉO
# ============================================================================

elif section == "🌦️ Météo & prévisions":

    st.subheader(
        "Météo en direct et prévisions à 7 jours"
    )

    source = (
        "en direct (Open-Meteo)"
        if live_ok
        else "estimation saisonnière (hors ligne)"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        f"Température actuelle — {source}",
        f"{ambient_live:.1f} °C",
    )

    c2.metric(
        "Humidité relative actuelle",
        f"{humidity_live:.0f} %",
    )

    rows, forecast_ok = (
        get_weekly_forecast()
    )

    st.caption(
        "Prévisions "
        + (
            "en direct (Open-Meteo)."
            if forecast_ok
            else
            "estimées hors ligne."
        )
    )

    df = pd.DataFrame(
        rows
    )

    df["Jour"] = (
        pd.to_datetime(
            df["date"]
        ).dt.strftime(
            "%a %d/%m"
        )
    )

    st.dataframe(
        df[
            [
                "Jour",
                "tmax",
                "tmin",
                "pluie_pct",
                "chaleur_pct",
            ]
        ].rename(
            columns={
                "tmax": "T° max (°C)",
                "tmin": "T° min (°C)",
                "pluie_pct": (
                    "Probabilité de pluie (%)"
                ),
                "chaleur_pct": (
                    "Indice de forte chaleur (%)"
                ),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        df,
        x="Jour",
        y=[
            "pluie_pct",
            "chaleur_pct",
        ],
        barmode="group",
        labels={
            "value": "%",
            "variable": "Indicateur",
        },
        title=(
            "Probabilité de pluie et indice "
            "de forte chaleur — 7 prochains jours"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# ============================================================================
# SECTION 6 — CARTE DU PARC
# ============================================================================

elif section == "🗺️ Carte du parc":

    st.subheader(
        "Carte du parc — état réel des transformateurs"
    )

    st.info(
        "La carte utilise le dernier état enregistré "
        "pour chaque transformateur. Par exemple, si TR-01 "
        "a atteint 60 % pendant une simulation, la carte "
        "conserve 60 % et affiche le repère en orange."
    )

    rows = []

    for t in TRANSFORMERS:

        status = (
            st.session_state
            .last_status[t["id"]]
        )

        risk = float(
            status["risk"]
        )

        band_label = status[
            "status"
        ]

        emoji = status[
            "emoji"
        ]

        source = status[
            "source"
        ]

        failure = status[
            "failure"
        ]

        if source == "simulation":

            if failure:

                etat = (
                    f"Simulation : "
                    f"{failure}"
                )

            else:

                etat = (
                    "Simulation — "
                    "fonctionnement normal"
                )

        elif source == "manuel":

            etat = (
                "Mode manuel"
            )

        else:

            etat = (
                "État initial"
            )

        rows.append(
            {
                "N°": t["id"],
                "Nom": t["nom"],
                "Type": t["type"],
                "Quartier": t[
                    "quartier"
                ],
                "Risque (%)": round(
                    risk,
                    1,
                ),
                "Statut": (
                    f"{emoji} "
                    f"{band_label}"
                ),
                "État": etat,
                "lat": t["lat"],
                "lon": t["lon"],
                "risk": risk,
            }
        )

    df_parc = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------------------
    # TABLEAU AVANT LA CARTE
    # ------------------------------------------------------------------------

    st.dataframe(
        df_parc[
            [
                "N°",
                "Nom",
                "Type",
                "Quartier",
                "Risque (%)",
                "Statut",
                "État",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------------
    # COULEURS
    # ------------------------------------------------------------------------

    def color_for(risk):

        if risk >= 65:

            # ROUGE
            return [
                220,
                40,
                40,
                240,
            ]

        elif risk >= 50:

            # ORANGE
            return [
                255,
                140,
                0,
                240,
            ]

        elif risk >= 35:

            # JAUNE
            return [
                245,
                200,
                40,
                240,
            ]

        else:

            # VERT
            return [
                40,
                170,
                70,
                240,
            ]

    df_parc["color"] = (
        df_parc["risk"]
        .apply(color_for)
    )

    # ------------------------------------------------------------------------
    # CARTE
    # ------------------------------------------------------------------------

    layer = pdk.Layer(
        "ScatterplotLayer",

        data=df_parc,

        get_position="[lon, lat]",

        get_fill_color="color",

        get_radius=35,

        radius_min_pixels=7,

        radius_max_pixels=16,

        stroked=True,

        get_line_color=[
            30,
            30,
            30,
            230,
        ],

        line_width_min_pixels=2,

        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=OUAGA_LAT,
        longitude=OUAGA_LON,
        zoom=13.5,
        pitch=0,
    )

    tooltip = {
        "text": (
            "{Nom} ({Type}) — {Quartier}\n"
            "Risque : {Risque (%) } % — "
            "{Statut}\n"
            "{État}"
        )
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_provider="carto",
            map_style="road",
        )
    )

    # ------------------------------------------------------------------------
    # LÉGENDE
    # ------------------------------------------------------------------------

    st.markdown(
        "### Légende du risque"
    )

    lc1, lc2, lc3, lc4 = st.columns(4)

    lc1.success(
        "🟢 **0–34 %** — Faible"
    )

    lc2.warning(
        "🟡 **35–49 %** — Modéré"
    )

    lc3.warning(
        "🟠 **50–64 %** — Interpellation"
    )

    lc4.error(
        "🔴 **65–100 %** — Élevé"
    )

    # ------------------------------------------------------------------------
    # OBSTACLES
    # ------------------------------------------------------------------------

    st.markdown(
        "### Obstacles et contraintes d'accès"
    )

    for t in TRANSFORMERS:

        with st.expander(
            f"{t['nom']} — "
            f"{t['quartier']} "
            f"({t['type']})"
        ):

            for obstacle in (
                t["obstacles"]
            ):

                st.markdown(
                    f"- {obstacle}"
                )

# ============================================================================
# MESSAGE GLOBAL
# ============================================================================

st.caption(
    "⚠️ Application de démonstration : les grandeurs "
    "électriques du mode manuel, les pannes du mode "
    "simulation et la météo simulée sont générées par "
    "l'application. La météo affichée en direct utilise "
    "Open-Meteo lorsque la connexion est disponible."
)

# ============================================================================
# ACTUALISATION TEMPS RÉEL
# ============================================================================

if st.session_state.running:

    time.sleep(1)

    st.rerun()

