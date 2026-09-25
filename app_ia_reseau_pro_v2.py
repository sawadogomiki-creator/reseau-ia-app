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
    page_title="IA Réseau Pro — Supervision intelligente",
    page_icon="⚡",
    layout="wide",
)

APP_VERSION = "2.0 — supervision intelligente & anticipation des pannes"

# ============================================================================
# INTERFACE DESKTOP — IA RÉSEAU PRO
# ============================================================================

st.markdown("""
<style>
:root { --app-bg:#eef3f8; --app-border:#cfd9e6; --app-text:#172033; --blue:#1261c9; --navy:#092b55; --cyan:#18a8d8; --green:#18a66b; --orange:#f59e0b; --red:#dc3545; }
.stApp { background:linear-gradient(135deg,#eef3f8 0%,#f8fbff 55%,#e8f1fb 100%); color:var(--app-text); }
.block-container { padding-top:5.8rem; padding-bottom:4rem; max-width:1540px; }
.desktop-titlebar { position:fixed; top:0; left:0; right:0; z-index:999; height:62px; background:linear-gradient(90deg,#061f43,#0b438d,#1685c9); color:white; display:flex; align-items:center; padding:0 26px; box-shadow:0 3px 16px rgba(8,43,85,.28); }
.desktop-brand { font-size:22px; font-weight:850; letter-spacing:.3px; }
.desktop-subtitle { margin-left:18px; font-size:13px; opacity:.82; border-left:1px solid rgba(255,255,255,.35); padding-left:18px; }
.toolbar { background:linear-gradient(180deg,#ffffff,#f3f7fb); border:1px solid var(--app-border); border-radius:12px; padding:8px 10px; margin:0 0 16px; box-shadow:0 3px 12px rgba(18,38,63,.07); }
.toolbar-title { font-size:11px; font-weight:800; color:#66809d; letter-spacing:1px; margin:0 0 5px 4px; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#fff,#f4f8fc); border:1px solid var(--app-border); border-radius:13px; padding:13px; box-shadow:0 3px 12px rgba(18,38,63,.07); }
.statusbar { position:fixed; bottom:0; left:0; right:0; z-index:998; background:#071d38; color:#dcecff; min-height:31px; display:flex; align-items:center; padding:0 16px; font-size:11px; box-shadow:0 -2px 10px rgba(0,0,0,.15); }
.statusbar span { margin-right:25px; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#071f3e,#0c315d 55%,#092646); border-right:1px solid #1e4773; }
section[data-testid="stSidebar"] * { color:#edf6ff; }
section[data-testid="stSidebar"] .stButton > button { text-align:left; border:1px solid transparent; background:rgba(255,255,255,.035); color:#eaf2ff; border-radius:8px; margin:2px 0; min-height:40px; padding-left:12px; }
section[data-testid="stSidebar"] .stButton > button:hover { background:rgba(33,156,221,.20); border-color:#3677aa; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color:#72c9ff; font-size:11px; letter-spacing:1.2px; }
.stButton > button { border-radius:8px; font-weight:650; }
[data-testid="stDataFrame"] { border:1px solid var(--app-border); border-radius:10px; overflow:hidden; box-shadow:0 2px 8px rgba(18,38,63,.05); }
.app-section { background:rgba(255,255,255,.72); border:1px solid #d7e1ec; border-radius:14px; padding:16px 18px; margin-bottom:16px; box-shadow:0 3px 12px rgba(18,38,63,.05); }
.section-banner { border-radius:12px; padding:13px 16px; color:white; background:linear-gradient(90deg,#0a448d,#1595ce); margin-bottom:15px; box-shadow:0 4px 12px rgba(10,68,141,.18); }
.section-banner h2 { margin:0; font-size:21px; }
.section-banner p { margin:3px 0 0; opacity:.86; font-size:12px; }
.report-box { background:linear-gradient(135deg,#eaf6ff,#f7fbff); border:1px solid #b9d9ef; border-left:5px solid #138bd1; border-radius:12px; padding:14px 16px; }
.page-header { background:linear-gradient(90deg,#0a448d,#18a8d8); color:white; border-radius:14px; padding:15px 20px; margin-bottom:18px; box-shadow:0 5px 16px rgba(10,68,141,.18); }
.page-header h2 { margin:0; color:white; }
.page-header p { margin:4px 0 0; opacity:.88; }
 .page-runtime-marker{display:none;}
/* IA RESEAU PRO 2.0 */
:root{--tech-navy:#061a33;--tech-blue:#0877d1;--tech-cyan:#15c5e8;--tech-green:#19b875;--tech-orange:#f6a623;--tech-red:#ef4656;}
html,body,.stApp{scroll-behavior:smooth;}
.stApp{background:radial-gradient(circle at 15% 10%,rgba(21,197,232,.10),transparent 26%),radial-gradient(circle at 85% 18%,rgba(8,119,209,.10),transparent 28%),linear-gradient(135deg,#edf4fa 0%,#f8fbfe 48%,#e7f0f8 100%);}
.block-container{max-width:1580px;padding-left:28px;padding-right:28px;}
.desktop-titlebar{height:66px;background:linear-gradient(105deg,#03152c 0%,#07386c 45%,#0877b8 100%);box-shadow:0 8px 28px rgba(3,35,68,.32);overflow:hidden;}
.desktop-titlebar:after{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);transform:translateX(-100%);animation:techSweep 5s linear infinite;}
.desktop-brand{font-size:23px;text-shadow:0 0 18px rgba(21,197,232,.35);}
.app-section,.toolbar,div[data-testid="stMetric"],.report-box{backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);}
.app-section{background:rgba(255,255,255,.78);border-color:rgba(148,177,204,.48);box-shadow:0 12px 35px rgba(12,50,83,.07);}
.page-header{position:relative;overflow:hidden;background:linear-gradient(110deg,#061e3e,#075fa8 58%,#10a8d5);box-shadow:0 12px 30px rgba(4,67,122,.20);}
.page-header:before{content:"";position:absolute;width:220px;height:220px;right:-55px;top:-115px;border:1px solid rgba(255,255,255,.20);border-radius:50%;box-shadow:0 0 0 22px rgba(255,255,255,.035),0 0 0 44px rgba(255,255,255,.025);}
.section-banner{background:linear-gradient(100deg,#061f40,#0878bd,#14b7d8);}
.stButton>button,.stDownloadButton>button{border:1px solid #b9cede;background:linear-gradient(180deg,#ffffff,#edf5fb);color:#0a3159;box-shadow:0 4px 12px rgba(7,48,83,.08);transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);border-color:#1a91cb;box-shadow:0 8px 20px rgba(8,119,209,.16);}
div[data-testid="stMetric"]{min-height:112px;position:relative;overflow:hidden;}
div[data-testid="stMetric"]:after{content:"";position:absolute;right:-25px;bottom:-35px;width:95px;height:95px;border:1px solid rgba(8,119,209,.10);border-radius:50%;box-shadow:0 0 0 16px rgba(8,119,209,.035);}
.tech-panel{background:linear-gradient(145deg,rgba(4,26,50,.97),rgba(7,65,105,.96));color:#eaf7ff;border:1px solid rgba(63,190,232,.25);border-radius:16px;padding:18px;box-shadow:0 12px 32px rgba(3,28,52,.20);position:relative;overflow:hidden;}
.tech-panel:after{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:linear-gradient(90deg,transparent,#15c5e8,#7be9ff,transparent);animation:techPulse 2.4s ease-in-out infinite;}
.tech-kicker{font-size:10px;letter-spacing:1.6px;color:#79d9f5;font-weight:800;text-transform:uppercase;}
.tech-value{font-size:28px;font-weight:850;line-height:1.1;margin-top:5px;}
.tech-muted{font-size:11px;color:#a9c8dc;}
.tech-chip{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);font-size:10px;font-weight:750;margin:3px 4px 0 0;}
.fleet-card{border:1px solid #cfdeea;border-radius:15px;padding:14px 15px;background:linear-gradient(145deg,#fff,#f2f7fb);box-shadow:0 8px 24px rgba(15,55,87,.07);transition:transform .2s ease,box-shadow .2s ease;}
.fleet-card:hover{transform:translateY(-3px);box-shadow:0 13px 30px rgba(15,55,87,.13);}
.fleet-name{font-weight:850;font-size:17px;color:#082b50;}
.fleet-risk{font-size:30px;font-weight:900;margin:5px 0;}
.fleet-line{height:6px;border-radius:99px;background:#dfe9f1;overflow:hidden;margin:7px 0;}
.fleet-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#18b87b,#f6a623,#ef4656);}
.live-dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:#18c77e;box-shadow:0 0 0 5px rgba(24,199,126,.12),0 0 14px rgba(24,199,126,.75);animation:liveBlink 1.5s ease-in-out infinite;}
.statusbar{height:32px;background:linear-gradient(90deg,#021326,#062b4e,#021326);}
@keyframes techSweep{0%{transform:translateX(-100%)}55%,100%{transform:translateX(100%)}}
@keyframes techPulse{0%,100%{opacity:.35}50%{opacity:1}}
@keyframes liveBlink{0%,100%{opacity:1}50%{opacity:.35}}
@media(max-width:900px){.desktop-subtitle{display:none}.block-container{padding-left:12px;padding-right:12px}}

.compact-head{display:flex;align-items:center;justify-content:space-between;gap:12px;background:linear-gradient(100deg,#061e3e,#0878bd);color:white;border-radius:12px;padding:10px 14px;margin-bottom:12px;}
.compact-head-title{font-size:17px;font-weight:850}.compact-head-sub{font-size:10px;opacity:.82;letter-spacing:.8px;text-transform:uppercase}
.control-panel{background:#f4f8fb;border:1px solid #d5e1eb;border-radius:12px;padding:10px;}
.control-title{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;font-weight:850;color:#5c7892;margin-bottom:6px}
.ai-panel{background:linear-gradient(145deg,#071d36,#0b426b);color:#edf8ff;border:1px solid #1a6a96;border-radius:12px;padding:12px;min-height:220px;}
.ai-big{font-size:38px;font-weight:900;line-height:1}.ai-small{font-size:10px;color:#a9cce0;text-transform:uppercase;letter-spacing:1px}.ai-line{height:1px;background:rgba(255,255,255,.12);margin:9px 0}
.ai-badge{display:inline-block;border-radius:999px;padding:4px 8px;font-size:10px;font-weight:800;background:rgba(255,255,255,.10);margin-top:5px}
.sim-strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px;margin-top:9px}
.sim-cell{background:#f7fafc;border:1px solid #d8e4ed;border-radius:9px;padding:7px 8px}.sim-label{font-size:9px;color:#70879b;text-transform:uppercase;letter-spacing:.7px}.sim-val{font-size:16px;font-weight:850;color:#0a3155}
.interpret-box{background:#edf8ff;border-left:4px solid #0b8ed0;border-radius:9px;padding:9px 11px;font-size:12px;color:#173a55}
@media(max-width:1100px){.sim-strip{grid-template-columns:repeat(3,minmax(0,1fr));}}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "modele_xgboost.pkl"

HISTORY_LEN = 120
ANIMATION_STEP = 0.18

OUAGA_LAT = 12.3714
OUAGA_LON = -1.5197

FAILURE_TIME_CONSTANT = 90.0
HEAT_ALERT_THRESHOLD = 38.0

# Simulation : True = à l'apparition de la panne, le risque part de 0 % puis
# rejoint progressivement la valeur calculée par le modèle (à 100 % de
# progression, le risque affiché est exactement celui du modèle).
# False = risque du modèle dès le départ (ne part pas de 0 %).
SIM_RELATIVE_RISK = True

# Grandeurs électriques en unités réelles (côté basse tension).
# ATTENTION : puissances nominales = VALEURS D'EXEMPLE, à remplacer par les
# valeurs des plaques signalétiques réelles des transformateurs.
RATINGS_KVA = {"TR-01": 630, "TR-02": 400, "TR-03": 160}
U_NOM = 400.0  # V, tension nominale BT entre phases (soit 230 V phase-neutre)


def nominal_current(t):
    """Courant nominal BT (A) : I = S / (√3 · U)."""
    return RATINGS_KVA.get(t["nom"], 400) * 1000 / (math.sqrt(3) * U_NOM)


def activate_manual(tid):
    """Appelé au premier réglage d'un curseur : lance le calcul du risque."""
    st.session_state.manual_active[tid] = True


def load_zone(c):
    if c < 0.8:
        return "🟢 charge légère"
    if c < 1.0:
        return "🟢 charge normale"
    if c < 1.2:
        return "🟠 surcharge admissible"
    return "🔴 surcharge critique"

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
    "☀️ Ciel dégagé (normal)": {"humidity_bonus": 0, "orage": False},
    "🌧️ Pluie": {"humidity_bonus": 30, "orage": False},
    "⛈️ Orage (pluie + foudre)": {"humidity_bonus": 35, "orage": True},
    "🌬️ Fraîcheur / harmattan frais": {"humidity_bonus": -15, "orage": False},
    "🔥 Canicule": {"humidity_bonus": -10, "orage": False},
}

SEASON_PRESETS = {
    "Saison sèche fraîche": {"temp_ref": 28, "humidity": 30},
    "Saison sèche chaude": {"temp_ref": 38, "humidity": 25},
    "Harmattan": {"temp_ref": 31, "humidity": 20},
    "Saison des pluies": {"temp_ref": 30, "humidity": 75},
}

TYPE_TIME_FACTOR = {
    "Cabine maçonnée": 1.35,
    "Préfabriqué": 1.0,
    "Haut de poteau": 0.7,
}

TYPE_CLIMATE = {
    "Cabine maçonnée": {"ambient_factor": 0.5, "humidity_factor": 0.35, "orage_voltage_bonus": 0.02},
    "Préfabriqué": {"ambient_factor": 0.85, "humidity_factor": 0.7, "orage_voltage_bonus": 0.05},
    "Haut de poteau": {"ambient_factor": 1.2, "humidity_factor": 1.0, "orage_voltage_bonus": 0.10},
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
    "🏠 Tableau de bord",
    "🖐️ Mode manuel",
    "🎲 Simulation de pannes",
    "📋 État des transformateurs",
    "📈 Historique des pannes",
    "🌦️ Météo & prévisions",
    "🗺️ Carte du parc",
]

SEASON_CODES = {
    "Saison sèche fraîche": 0,
    "Saison sèche chaude": 1,
    "Harmattan": 2,
    "Saison des pluies": 3,
}

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


def _model_input(f: dict, season_code: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "charge": f["charge"],
        "oil": f["oil_temp"],
        "ambient": f["ambient"],
        "humidity": f["humidity"],
        "voltage": f["voltage"],
        "season": season_code,
    }])


def measurement_boundary_risk(features: dict) -> tuple[float, list[str]]:
    """Détecte les valeurs proches des limites de mesure.

    Un niveau très faible n'est pas assimilé directement à une panne : il peut
    signaler un capteur, une acquisition ou un transformateur réellement hors
    service. Le score reste volontairement modéré tant qu'aucune autre grandeur
    ne confirme une dégradation.
    """
    risk = 0.0
    alerts = []
    voltage = float(features.get("voltage", 1.0))
    charge = float(features.get("charge", 0.8))
    oil = float(features.get("oil_temp", 50.0))
    humidity = float(features.get("humidity", 50.0))

    if voltage <= 0.92:
        risk += min(14.0, (0.92 - voltage) * 140.0)
        alerts.append("Tension très basse : vérifier réseau, mesure et capteur.")
    elif voltage >= 1.08:
        risk += min(18.0, (voltage - 1.08) * 180.0)
        alerts.append("Tension très élevée : vérifier réseau et protection contre les surtensions.")

    if charge <= 0.05:
        risk += 4.0
        alerts.append("Courant presque nul : vérifier capteur, départ BT et état du transformateur.")
    elif charge >= 1.25:
        risk += min(22.0, (charge - 1.25) * 70.0)
        alerts.append("Charge très élevée : risque de surcharge thermique.")

    if oil <= 35.5:
        risk += 3.0
        alerts.append("Température huile anormalement basse : vérifier capteur et cohérence de la mesure.")
    elif oil >= 95.0:
        risk += min(20.0, (oil - 95.0) * 0.8)
        alerts.append("Température huile très élevée : vérifier échauffement et refroidissement.")

    if humidity <= 8.0:
        risk += 1.5
        alerts.append("Humidité très faible : vérifier la cohérence du capteur avant interprétation.")
    elif humidity >= 90.0:
        risk += min(12.0, (humidity - 90.0) * 0.6)
        alerts.append("Humidité très élevée : contrôler l'isolement et les conditions ambiantes.")

    return float(np.clip(risk, 0.0, 35.0)), alerts


def responsive_operational_risk(features: dict) -> float:
    """Composante physique réactive pour éviter un retard visuel du risque.

    Le modèle XGBoost reste la composante IA. Cette couche traduit rapidement
    les dépassements évidents des grandeurs surveillées afin que l'interface
    réagisse dès qu'un curseur entre dans une zone préoccupante.
    """
    charge = float(features.get("charge", 0.8))
    oil = float(features.get("oil_temp", 50.0))
    ambient = float(features.get("ambient", 32.0))
    humidity = float(features.get("humidity", 50.0))
    voltage = float(features.get("voltage", 1.0))

    risk = 1.2  # risque résiduel minimal en fonctionnement normal

    # Charge : réaction volontairement rapide après 80 % de charge.
    if charge > 0.80:
        risk += (charge - 0.80) * 115.0

    # Température huile : accélération progressive puis forte au-delà de 75 °C.
    if oil > 60.0:
        risk += (oil - 60.0) * 1.35
    if oil > 75.0:
        risk += (oil - 75.0) * 1.00

    # Ambiance chaude : effet secondaire mais visible.
    if ambient > 32.0:
        risk += (ambient - 32.0) * 1.8

    # Humidité élevée : contribution à l'isolement.
    if humidity > 60.0:
        risk += (humidity - 60.0) * 0.55

    # Écart de tension : réaction des deux côtés de la valeur nominale.
    deviation = abs(voltage - 1.0)
    if deviation > 0.02:
        risk += (deviation - 0.02) * 105.0

    return float(np.clip(risk, 1.0, 100.0))


def predict_risk(features: dict, ttype: str) -> float:
    season_code = SEASON_CODES.get(features.get("season_label", "Saison sèche chaude"), 1)

    if MODEL is not None:
        try:
            model_risk = float(MODEL.predict_proba(_model_input(features, season_code))[0][1] * 100)
        except Exception:
            model_risk = _heuristic_risk(features, season_code)
    else:
        model_risk = _heuristic_risk(features, season_code)

    # Couche réactive : elle permet au risque affiché de suivre immédiatement
    # une variation physique, même si l'arbre XGBoost change peu localement.
    reactive = responsive_operational_risk(features)
    boundary_risk, _ = measurement_boundary_risk(features)
    model_component = model_risk * TYPE_MULTIPLIER.get(ttype, 1.0)

    # Le modèle conserve le poids principal ; la couche réactive garantit une
    # réponse rapide aux dépassements évidents.
    combined = 0.65 * model_component + 0.35 * reactive + boundary_risk

    # Un fonctionnement normal n'est jamais affiché à 0 %.
    floor = 1.0
    return float(np.clip(max(floor, combined), floor, 100))


def _heuristic_risk(f: dict, season_code: int):
    charge_term = max(0.0, f["charge"] - 0.8) * 55
    oil_term = max(0.0, f["oil_temp"] - 65) * 1.6
    ambient_term = max(0.0, f["ambient"] - 32) * 1.2
    humidity_term = max(0.0, f["humidity"] - 60) * 0.5
    voltage_term = abs(f["voltage"] - 1.0) * 60
    season_term = {0: 0, 1: 8, 2: 5, 3: 10}.get(season_code, 0)

    total = (
        8 + charge_term + oil_term + ambient_term
        + humidity_term + voltage_term + season_term
    )
    return float(np.clip(total, 0, 100))


# ============================================================================
# PRIORISATION OPÉRATIONNELLE
# ============================================================================
# Ces paramètres décrivent uniquement la criticité opérationnelle du prototype.
# Ils ne remplacent pas une criticité métier SONABEL validée sur le terrain.

CRITICALITY = {
    "TR-01": {"criticite": 0.45, "acces": 0.95, "impact_reseau": 0.70},
    "TR-02": {"criticite": 0.65, "acces": 0.60, "impact_reseau": 0.85},
    "TR-03": {"criticite": 0.85, "acces": 0.35, "impact_reseau": 0.75},
}


def transformer_priority(t, risk):
    """Calcule une priorité d'intervention indicative pour le prototype."""
    c = CRITICALITY.get(
        t["nom"],
        {"criticite": 0.50, "acces": 0.50, "impact_reseau": 0.50},
    )
    # Risque = moteur principal ; criticité réseau et accessibilité modulent la priorité.
    raw = (
        0.60 * float(risk)
        + 100 * 0.20 * c["criticite"]
        + 100 * 0.15 * c["impact_reseau"]
        + 100 * 0.05 * (1 - c["acces"])
    )
    raw = float(np.clip(raw, 0, 100))

    if raw >= 80:
        return "P1 — Intervention immédiate", raw
    if raw >= 65:
        return "P2 — Intervention prioritaire", raw
    if raw >= 45:
        return "P3 — Surveillance renforcée", raw
    return "P4 — Surveillance normale", raw


def local_risk_sensitivity(features, ttype):
    """
    Explication locale légère et robuste.
    Si XGBoost est disponible, on mesure l'effet d'une petite variation
    de chaque variable. Sinon, on expose les composantes de l'heuristique.
    Ce n'est PAS une valeur SHAP.
    """
    feature_defs = [
        ("charge", "Charge relative", 0.05),
        ("oil_temp", "Température huile", 2.0),
        ("ambient", "Température ambiante", 2.0),
        ("humidity", "Humidité", 5.0),
        ("voltage", "Tension", 0.02),
    ]

    season_code = SEASON_CODES.get(features.get("season_label", "Saison sèche chaude"), 1)

    if MODEL is not None:
        try:
            def model_value(f):
                return float(MODEL.predict_proba(_model_input(f, season_code))[0][1] * 100)

            base = model_value(features)

            rows = []
            for key, label, delta in feature_defs:
                plus = dict(features)
                minus = dict(features)
                plus[key] = float(plus[key]) + delta
                minus[key] = float(minus[key]) - delta

                sensitivity = (model_value(plus) - model_value(minus)) / 2.0

                rows.append({
                    "Variable": label,
                    "Effet local (pts)": round(sensitivity, 2),
                    "Lecture": (
                        "augmente le risque" if sensitivity > 0.05
                        else "réduit le risque" if sensitivity < -0.05
                        else "effet faible"
                    ),
                })

            df = pd.DataFrame(rows)
            df["Valeur actuelle"] = [round(float(features[k]), 3) for k, _, _ in feature_defs]
            df.attrs["source"] = "XGBoost — sensibilité locale"
            df.attrs["base"] = base
            return df
        except Exception:
            pass

    # Fallback explicable : composantes de l'heuristique.
    values = [
        max(0, features["charge"] - 0.8) * 55,
        max(0, features["oil_temp"] - 65) * 1.6,
        max(0, features["ambient"] - 32) * 1.2,
        max(0, features["humidity"] - 60) * 0.5,
        abs(features["voltage"] - 1.0) * 60,
    ]
    labels = [x[1] for x in feature_defs]
    vals = [features[x[0]] for x in feature_defs]
    df = pd.DataFrame({
        "Variable": labels,
        "Effet local (pts)": np.round(values, 2),
        "Valeur actuelle": np.round(vals, 3),
        "Lecture": [
            "contribue au risque" if v > 0.05 else "effet faible"
            for v in values
        ],
    })
    df.attrs["source"] = "Heuristique — composantes du risque"
    return df


def render_temperature_gauge(temp, label="Température ambiante"):
    value = float(np.clip(temp, 15, 50))
    pct = (value - 15) / 35
    if value < 30:
        state, icon = "Zone fraîche", "🟢"
    elif value < 35:
        state, icon = "Zone normale", "🟡"
    elif value < 40:
        state, icon = "Chaleur élevée", "🟠"
    else:
        state, icon = "Très forte chaleur", "🔴"
    st.markdown(f"**🌡️ {label} : {value:.1f} °C** · {icon} {state}")
    st.progress(pct, text=f"15 °C    ─────────────    50 °C   |   {value:.1f} °C")


# ============================================================================
# DÉCISION ASSISTÉE PAR L'IA
# ============================================================================
# Répartition des rôles (à expliquer au jury) :
#   - le modèle XGBoost calcule le risque, identifie la variable qui pèse le
#     plus dans ce risque, et estime l'effet d'une action corrective ;
#   - les règles métier (dictionnaires ci-dessous) traduisent cette cause en
#     démarche concrète à engager.

DRIVER_ACTIONS = {
    "Charge relative": (
        "Réduire la charge : reporter une partie des abonnés vers un poste "
        "voisin ou délester les départs les moins prioritaires."
    ),
    "Température huile": (
        "Contrôler le refroidissement (radiateurs, niveau d'huile) et réduire "
        "la charge le temps que l'huile refroidisse."
    ),
    "Température ambiante": (
        "Limiter la charge aux heures les plus chaudes et renforcer la "
        "surveillance thermique de ce poste."
    ),
    "Humidité": (
        "Contrôler l'étanchéité et l'assécheur d'air, puis mesurer la "
        "résistance d'isolement."
    ),
    "Tension": (
        "Vérifier le réglage de la tension amont et l'état des parafoudres."
    ),
}

RISK_LEVEL_ACTIONS = [
    (65, "🔴 Inspection prioritaire"),
    (50, "🟠 Planifier une inspection préventive"),
    (35, "🟡 Surveillance renforcée"),
    (0, "🟢 Surveillance normale"),
]


# ============================================================================
# ANTICIPATION DU TYPE DE PANNE
# ============================================================================
# Le modèle XGBoost donne un risque GLOBAL de panne. La répartition de ce
# risque entre les types de panne est estimée en comparant l'écart observé
# (par rapport à un fonctionnement sain) aux trajectoires des 5 pannes
# connues (FAILURE_MODES) : plus l'écart ressemble à une panne, plus elle
# reçoit une part du risque global. Ce n'est PAS une probabilité issue d'un
# modèle multi-classe ; pour cela il faudrait entraîner le modèle avec le
# type de panne comme étiquette.

FAILURE_LABELS = {
    "Surtension (foudre / manœuvre)": "surtension",
    "Court-circuit interne": "court-circuit",
    "Température élevée / surcharge thermique": "surchauffe",
    "Surcharge électrique prolongée": "surcharge",
    "Défaut d'isolement (humidité)": "défaut d'isolement",
}

SIGNATURE_SCALES = {"charge": 0.4, "oil_temp": 15.0, "voltage": 0.2, "humidity": 25.0, "ambient": 5.0}
SIGNATURE_POSITIVE = {"charge", "oil_temp", "humidity", "ambient"}  # nocif quand ça monte
SIGNATURE_SIGMA = 0.4  # tolérance d'écart de direction


def default_reference(features):
    """Fonctionnement sain de référence (mode manuel)."""
    season = SEASON_PRESETS.get(
        features.get("season_label"), SEASON_PRESETS["Saison sèche chaude"]
    )
    return {
        "charge": 0.8,
        "oil_temp": 45 + 28 * 0.8,
        "voltage": 1.0,
        "humidity": float(season["humidity"]),
        "ambient": 35.0,
    }


def failure_shares(features, reference=None):
    """Part (0..1, somme = 1) attribuée à chaque type de panne."""
    ref = reference or default_reference(features)
    dev = {}
    for k, sc in SIGNATURE_SCALES.items():
        d = float(features[k]) - float(ref[k])
        if k in SIGNATURE_POSITIVE:
            d = max(0.0, d)
        dev[k] = d / sc

    nd2 = sum(v * v for v in dev.values())
    if nd2 < 0.15 ** 2:
        return {}  # aucun écart notable : pas de signature de panne

    scores = {}
    for name, fm in FAILURE_MODES.items():
        e = {k: fm["effet"].get(k, 0.0) / sc for k, sc in SIGNATURE_SCALES.items()}
        ne2 = sum(v * v for v in e.values())
        dot = sum(dev[k] * e[k] for k in SIGNATURE_SCALES)
        if dot <= 0:
            scores[name] = 0.0
            continue
        progress = dot / ne2                                   # ampleur le long de la panne
        resid = math.sqrt(max(0.0, nd2 - dot * dot / ne2) / nd2)  # écart de direction
        scores[name] = min(1.0, progress) * math.exp(-((resid / SIGNATURE_SIGMA) ** 2))

    total = sum(scores.values())
    if total <= 1e-9:
        return {}
    return {n: v / total for n, v in scores.items()}


def anticipated_failures(features, global_risk, reference=None):
    """Liste triée (nom, libellé, risque %) : part du risque global par type de panne."""
    shares = failure_shares(features, reference)
    rows = [
        (name, FAILURE_LABELS.get(name, name), float(global_risk) * w)
        for name, w in shares.items()
    ]
    rows.sort(key=lambda r: -r[2])
    return rows


def render_failure_forecast(features, risk, slope, reference=None, simulated=None):
    """Affiche la panne anticipée : « Risque de court-circuit 20 % »."""
    rows = anticipated_failures(features, risk, reference)
    top = rows[0] if rows and rows[0][2] >= 1.0 else None

    if top:
        st.metric(f"Risque de {top[1]}", f"{top[2]:.0f} %")
    else:
        st.metric("Panne anticipée", "Aucune")

    st.caption(f"Risque global : **{risk:.1f} %** · tendance {slope:+.2f} pt/min")

    shown = [r for r in rows[:3] if r[2] >= 1.0]
    for _, label, pct in shown:
        st.progress(min(1.0, pct / 100), text=f"{label.capitalize()} — {pct:.0f} %")
    if shown:
        st.caption("Répartition estimée du risque global par type de panne.")

    if simulated:
        sim_label = FAILURE_LABELS.get(simulated, simulated)
        if top and top[0] == simulated:
            st.caption(f"🎯 Scénario simulé : {sim_label} — identifié par l'IA.")
        else:
            st.caption(f"Scénario simulé : {sim_label} — signature encore ambiguë.")


def ai_what_if(features, ttype, driver, bonus=0.0):
    """Risque recalculé par le modèle si l'on agit sur la cause principale."""
    f = dict(features)
    if driver in ("Charge relative", "Température huile", "Température ambiante"):
        d = min(0.2, f["charge"] - 0.2)
        if d <= 0:
            return None
        f["charge"] -= d
        f["oil_temp"] -= 28 * d  # même relation charge → huile que le mode manuel
        label = f"charge réduite de {d * 100:.0f} points"
    elif driver == "Tension":
        f["voltage"] = 1.0
        label = "tension ramenée à 1,00 p.u."
    else:
        return None
    after = float(np.clip(predict_risk(f, ttype) + bonus, 0, 100))
    return after, label


def ai_decision(features, ttype, risk, bonus=0.0, reference=None):
    df = local_risk_sensitivity(features, ttype)
    top = df.sort_values("Effet local (pts)", ascending=False).iloc[0]
    effect = float(top["Effet local (pts)"])
    driver = top["Variable"] if effect > 0.05 else None
    level_action = next(txt for lo, txt in RISK_LEVEL_ACTIONS if risk >= lo)
    what_if = ai_what_if(features, ttype, driver, bonus) if (driver and risk >= 35) else None
    failures = anticipated_failures(features, risk, reference)
    failure = failures[0] if failures and failures[0][2] >= 5.0 else None
    return {
        "failure": failure,
        "driver": driver,
        "effect": effect,
        "source": df.attrs.get("source", "analyse locale"),
        "level_action": level_action,
        "action": DRIVER_ACTIONS.get(driver) if driver else None,
        "what_if": what_if,
    }


def render_ai_card(t, features, risk, mode, bonus=0.0, reference=None):
    """Fiche de décision : diagnostic, cause, démarche, effet attendu."""
    d = ai_decision(features, t["type"], risk, bonus, reference)
    priority, score = transformer_priority(t, risk)

    # Notification uniquement lors d'un changement de niveau.
    band, _ = risk_band(risk)
    key = f"prev_band_{mode}_{t['id']}"
    prev = st.session_state.get(key)
    if prev is not None and prev != band and risk >= 50:
        st.toast(f"{t['nom']} : risque passé en « {band} » ({risk:.0f} %)", icon="🚨")
    st.session_state[key] = band

    with st.container(border=True):
        st.markdown("**🤖 Décision assistée par l'IA**")
        st.markdown(f"**Diagnostic :** risque {risk:.1f} % — {d['level_action']}")
        if d["driver"]:
            st.markdown(
                f"**Cause principale détectée :** {d['driver']} "
                f"(sensibilité locale {d['effect']:+.1f} pt)"
            )
        if d["failure"] and risk >= 35:
            name, label, pct = d["failure"]
            st.markdown(f"**Panne à anticiper :** {label} (≈ {pct:.0f} %)")
            steps = (
                NETWORK_ACTIONS_BY_FAILURE.get(name, [])[:1]
                + FAILURE_MODES[name]["directives"][:1]
            )
            st.markdown(
                "**Démarche recommandée :**\n"
                + "\n".join(f"- {step}" for step in steps)
            )
        elif d["action"] and risk >= 35:
            st.markdown(f"**Démarche recommandée :** {d['action']}")
        if d["what_if"]:
            after, label = d["what_if"]
            st.markdown(
                f"**Effet attendu :** {label} → risque "
                f"**{risk:.1f} % → {after:.1f} %** ({after - risk:+.1f} pts)"
            )
        st.caption(
            f"Priorité indicative : {priority} ({score:.0f}/100) · "
            f"Source : {d['source']}. Estimation à confirmer par les "
            "procédures d'exploitation."
        )


def season_for_month(month):
    if month in (3, 4, 5):
        return "Saison sèche chaude"
    if month in (6, 7, 8, 9, 10):
        return "Saison des pluies"
    return "Harmattan"


def render_ai_forecast(rows):
    """Anticipation : risque prévu par le modèle à partir des prévisions météo."""
    st.markdown("### 🔮 Risque prévu par l'IA sur 7 jours")
    st.caption(
        "Le modèle est appliqué aux prévisions météo des 7 prochains jours. "
        "Hypothèses : charge constante (curseur ci-dessous), tension nominale, "
        "humidité déduite de la probabilité de pluie, température d'huile "
        "calculée comme en mode manuel."
    )
    charge = st.slider(
        "Charge moyenne supposée (relative)", 0.4, 1.3, 0.9, 0.05,
        key="forecast_charge",
    )
    season = season_for_month(datetime.now().month)

    records = []
    for r in rows:
        day = pd.to_datetime(r["date"]).strftime("%a %d/%m")
        hum = float(np.clip(25 + 0.55 * r["pluie_pct"], 5, 100))
        oil = 45 + 28 * charge + max(0.0, r["tmax"] - 30) * 0.6
        for t in TRANSFORMERS:
            f = {
                "charge": charge,
                "oil_temp": oil,
                "ambient": r["tmax"],
                "humidity": hum,
                "voltage": 1.0,
                "season_label": season,
            }
            records.append({
                "Jour": day,
                "Transformateur": t["nom"],
                "Risque prévu (%)": round(predict_risk(f, t["type"]), 1),
            })

    df = pd.DataFrame(records)

    fig = px.line(df, x="Jour", y="Risque prévu (%)", color="Transformateur", markers=True)
    fig.update_yaxes(range=[0, 100])
    fig.add_hline(y=50, line_dash="dash", line_color="orange")
    fig.add_hline(y=65, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True, key="forecast_risk_chart")

    watch = df[df["Risque prévu (%)"] >= 50]
    if watch.empty:
        st.success(
            "🟢 Aucun transformateur ne dépasse 50 % de risque sur les 7 "
            "prochains jours avec ces hypothèses."
        )
    else:
        worst = watch.sort_values("Risque prévu (%)", ascending=False).iloc[0]
        st.warning(
            f"⚠️ L'IA anticipe {len(watch)} situation(s) à 50 % ou plus. "
            f"Pic : **{worst['Transformateur']}** le **{worst['Jour']}** "
            f"({worst['Risque prévu (%)']:.1f} %). Démarche : programmer une "
            "inspection préventive avant cette date et préparer un report de charge."
        )
        st.dataframe(watch, use_container_width=True, hide_index=True)


def build_intervention_report():
    """Rapport détaillé fondé exclusivement sur l'état du mode simulation."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    lines = [
        "RAPPORT COMPLET DE SIMULATION ET CONDUITE À TENIR",
        "IA RÉSEAU PRO — PARC DE TRANSFORMATEURS SONABEL",
        f"Date de génération : {now}",
        "=" * 78,
        "OBJET DU RAPPORT",
        "Ce rapport présente les résultats de la simulation de panne et les opérations",
        "à envisager selon le scénario simulé. Il s'agit d'un outil de démonstration :",
        "les décisions réelles doivent être confirmées par les procédures d'exploitation",
        "et les mesures terrain.",
        "",
    ]
    any_failure = False
    for t in TRANSFORMERS:
        status = st.session_state.simulation_status[t["id"]]
        risk = float(status.get("risk", 0))
        failure = status.get("failure")
        progress = float(status.get("progress", 0))
        priority, priority_score = transformer_priority(t, risk)
        features = status.get("features", {})
        if failure:
            any_failure = True
        lines += [
            "-" * 78,
            f"{t['nom']} — {t['type']} — {t['quartier']}",
            f"Scénario simulé : {failure or 'Fonctionnement normal'}",
            f"Niveau de risque simulé : {risk:.1f} % — {status.get('status', 'Faible')}",
            f"Progression de la panne : {progress * 100:.1f} %",
            f"Tendance simulée : {status.get('trend_cat', 'stable')} ({status.get('trend', 0):+.2f} pt/min)",
            f"Priorité indicative : {priority} — score {priority_score:.1f}/100",
        ]
        if features:
            top_af = anticipated_failures(features, risk, status.get("reference"))
            if top_af and top_af[0][2] >= 1.0:
                lines.append(
                    f"Panne anticipée par l'IA : {top_af[0][1]} ({top_af[0][2]:.0f} %)"
                )
            lines += [
                "Grandeurs simulées :",
                f"  • Charge relative : {features.get('charge', 0):.2f}",
                f"  • Température huile : {features.get('oil_temp', 0):.1f} °C",
                f"  • Température ambiante : {features.get('ambient', 0):.1f} °C",
                f"  • Humidité : {features.get('humidity', 0):.1f} %",
                f"  • Tension : {features.get('voltage', 0):.3f} p.u.",
            ]
        if failure:
            phase_label, _, phase_text = failure_phase(progress)
            lines += [
                f"Phase : {phase_label}",
                f"Interprétation : {phase_text}",
                "Conduite à tenir / opérations recommandées :",
            ]
            for action in NETWORK_ACTIONS_BY_FAILURE.get(failure, []):
                lines.append(f"  • {action}")
            lines.append(f"  • {NETWORK_ACTIONS_BY_TYPE.get(t['type'], '')}")
            lines.append("Directives techniques ciblées :")
            for directive in FAILURE_MODES[failure]["directives"]:
                lines.append(f"  • {directive}")
            lines += [
                "Commentaire opérationnel :",
                f"  Le scénario « {failure} » nécessite une surveillance adaptée au niveau",
                "  de risque obtenu. Plus la progression est élevée, plus la préparation",
                "  de l'intervention doit être anticipée. Toute consignation, manœuvre ou",
                "  réalimentation doit respecter les procédures de sécurité applicables.",
            ]
        else:
            lines += [
                "Conduite à tenir :",
                "  • Maintenir la surveillance des grandeurs électriques et thermiques.",
                "  • Conserver une vigilance sur l'évolution du risque simulé.",
                "Commentaire : aucun scénario de panne actif pour ce transformateur.",
            ]
        lines.append("")
    lines += ["=" * 78, "SYNTHÈSE DE CONDUITE À TENIR"]
    if any_failure:
        lines += [
            "Au moins un scénario de panne est actif dans la simulation.",
            "Les opérations recommandées sont celles associées aux scénarios sélectionnés.",
            "En situation réelle, confirmer les alarmes par les mesures disponibles,",
            "sécuriser la zone, appliquer les procédures de consignation et coordonner",
            "l'intervention avec le centre de conduite avant toute manœuvre réseau.",
        ]
    else:
        lines += [
            "Aucune panne n'est actuellement sélectionnée dans la simulation.",
            "Le parc peut être observé en fonctionnement normal dans le cadre du prototype.",
        ]
    return "\n".join(lines)


# ============================================================================
# ÉTAT DE SESSION
# ============================================================================

def empty_status():
    return {
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


def create_last_status():
    return {t["id"]: empty_status() for t in TRANSFORMERS}


def _manual_defaults():
    return {"voltage": 1.0, "charge": 0.0, "current_ratio": 0.0}


def init_state():
    defaults = {
        "running": True,
        "section": SECTIONS[0],
        "history": lambda: {t["id"]: deque(maxlen=HISTORY_LEN) for t in TRANSFORMERS},
        # Données strictement internes au mode simulation.
        # Elles ne sont jamais utilisées par le tableau de bord,
        # le mode manuel, l'état des transformateurs ou la carte.
        "simulation_history": lambda: {t["id"]: deque(maxlen=HISTORY_LEN) for t in TRANSFORMERS},
        "manual_target": lambda: {t["id"]: _manual_defaults() for t in TRANSFORMERS},
        "manual_effective": lambda: {t["id"]: _manual_defaults() for t in TRANSFORMERS},
        "manual_active": lambda: {t["id"]: False for t in TRANSFORMERS},
        "manual_temperature": 32.0,
        "manual_season": "Saison sèche chaude",
        "simulation_temperature": 32.0,
        "simulation_season": "Saison sèche chaude",
        "sim_failure": lambda: {t["id"]: None for t in TRANSFORMERS},
        "sim_start_time": lambda: {t["id"]: None for t in TRANSFORMERS},
        "event_log": list,
        "active_event": lambda: {t["id"]: None for t in TRANSFORMERS},
        # État opérationnel utilisé par les pages hors simulation.
        "last_status": create_last_status,
        # État strictement isolé du mode simulation.
        "simulation_status": create_last_status,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default() if callable(default) else default

    # Sécurité pour les anciens états de session
    for t in TRANSFORMERS:
        tid = t["id"]
        st.session_state.manual_target.setdefault(tid, _manual_defaults())
        st.session_state.manual_effective.setdefault(tid, _manual_defaults())
        st.session_state.manual_active.setdefault(tid, False)
        st.session_state.sim_failure.setdefault(tid, None)
        st.session_state.sim_start_time.setdefault(tid, None)
        st.session_state.active_event.setdefault(tid, None)
        st.session_state.last_status.setdefault(tid, empty_status())
        st.session_state.simulation_status.setdefault(tid, empty_status())
        if tid not in st.session_state.history:
            st.session_state.history[tid] = deque(maxlen=HISTORY_LEN)
        if tid not in st.session_state.simulation_history:
            st.session_state.simulation_history[tid] = deque(maxlen=HISTORY_LEN)


# ============================================================================
# HISTORIQUE
# ============================================================================

def _trend_from(hist):
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


def push_history(t_id, risk_pct, features):
    st.session_state.history[t_id].append({"t": datetime.now(), "risk": risk_pct, **features})


def compute_trend(t_id):
    return _trend_from(st.session_state.history[t_id])


def push_simulation_history(t_id, risk_pct, features):
    """Historique privé au mode simulation."""
    st.session_state.simulation_history[t_id].append({"t": datetime.now(), "risk": risk_pct, **features})


def compute_simulation_trend(t_id):
    """Tendance calculée uniquement sur l'historique simulé."""
    return _trend_from(st.session_state.simulation_history[t_id])


def _status_dict(risk, slope, trend_cat, band_label, emoji, source, failure, progress, features):
    return {
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


def update_simulation_status(
    t_id, risk, slope, trend_cat, band_label, emoji,
    failure=None, progress=0.0, features=None, source="Simulation + XGBoost",
):
    """Met à jour uniquement l'état interne de la simulation."""
    st.session_state.simulation_status[t_id] = _status_dict(
        risk, slope, trend_cat, band_label, emoji, source, failure, progress, features
    )


def update_last_status(
    t_id, risk, slope, trend_cat, band_label, emoji, source,
    failure=None, progress=0.0, features=None,
):
    st.session_state.last_status[t_id] = _status_dict(
        risk, slope, trend_cat, band_label, emoji, source, failure, progress, features
    )


# ============================================================================
# MANUEL
# ============================================================================

def animate_towards_target(t_id):
    tgt = st.session_state.manual_target[t_id]
    eff = st.session_state.manual_effective[t_id]

    for k in tgt:
        eff[k] += (tgt[k] - eff[k]) * ANIMATION_STEP
        if abs(eff[k] - tgt[k]) < 1e-3:
            eff[k] = tgt[k]


# ============================================================================
# ÉVÉNEMENTS
# ============================================================================

def log_event_if_needed(t_id, nom, band_label, risk, panne_label):
    active = st.session_state.active_event[t_id]

    if band_label == "Élevé":
        if active is None:
            st.session_state.event_log.append({
                "transfo_id": t_id,
                "nom": nom,
                "type_panne": panne_label or "Mode manuel",
                "debut": datetime.now(),
                "fin": None,
                "risque_max": risk,
            })
            st.session_state.active_event[t_id] = len(st.session_state.event_log) - 1
        else:
            ev = st.session_state.event_log[active]
            ev["risque_max"] = max(ev["risque_max"], risk)
    else:
        if active is not None:
            st.session_state.event_log[active]["fin"] = datetime.now()
            st.session_state.active_event[t_id] = None


# ============================================================================
# PROGRESSION PANNE
# ============================================================================

def get_failure_progress(t_id, ttype):
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
# PHASE DE PANNE
# ============================================================================

def failure_phase(progress):
    if progress < 0.12:
        return (
            "🔍 Anomalie naissante",
            "info",
            "Un écart encore ténu vient d'apparaître sur ce transformateur.",
        )
    if progress < 0.45:
        return (
            "⚠️ Dégradation en cours",
            "warning",
            "L'écart se creuse progressivement. Une intervention préventive peut limiter l'aggravation.",
        )
    if progress < 0.8:
        return (
            "🟠 Aggravation avancée",
            "warning",
            "La dégradation s'accélère. L'état critique approche.",
        )
    return (
        "🚨 État critique",
        "error",
        "Le transformateur approche d'un état de défaillance critique.",
    )


# ============================================================================
# PRONOSTIC
# ============================================================================

def prognosis_text(risk_pct, slope, failure_label=None):
    label = failure_label or "défaillance"

    if slope <= 0.15:
        return (
            f"Évolution stable — pas d'aggravation "
            f"significative détectée pour {label.lower()}."
        )

    remaining_pct = max(0.0, 90 - risk_pct)
    minutes_to_90 = remaining_pct / slope if slope > 0 else None

    if minutes_to_90 is None or minutes_to_90 > 24 * 60:
        return f"Tendance à la hausse lente pour {label.lower()} — surveillance renforcée."

    hours = minutes_to_90 / 60
    delay = f"environ {int(minutes_to_90)} min" if hours < 1 else f"environ {hours:.1f} h"

    return (
        f"Extrapolation : le seuil de 90 % pourrait être atteint "
        f"dans {delay} si la pente actuelle reste inchangée. "
        f"Ce n'est pas une estimation statistique du temps réel avant panne."
    )


# ============================================================================
# MÉTÉO LIVE
# ============================================================================

@st.cache_data(ttl=600)
def get_live_weather():
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={OUAGA_LAT}&longitude={OUAGA_LON}"
            f"&current=temperature_2m,relative_humidity_2m"
            f"&timezone=Africa%2FOuagadougou"
        )
        data = requests.get(url, timeout=4).json()
        return (
            float(data["current"]["temperature_2m"]),
            float(data["current"]["relative_humidity_2m"]),
            True,
        )
    except Exception:
        month = datetime.now().month
        temp = 38 if month in (3, 4, 5) else (28 if month in (7, 8, 9) else 33)
        hum = 75 if month in (7, 8, 9) else 25
        return float(temp), float(hum), False


# ============================================================================
# PRÉVISIONS 7 JOURS
# ============================================================================

@st.cache_data(ttl=1800)
def get_weekly_forecast():
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={OUAGA_LAT}&longitude={OUAGA_LON}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            f"&forecast_days=7&timezone=Africa%2FOuagadougou"
        )
        data = requests.get(url, timeout=5).json()
        d = data["daily"]

        rows = []
        for i, date in enumerate(d["time"]):
            tmax = float(d["temperature_2m_max"][i])
            tmin = float(d["temperature_2m_min"][i])
            rain_pct = float(d["precipitation_probability_max"][i])
            heat_pct = float(np.clip((tmax - 32) / 13 * 100, 0, 100))
            rows.append({
                "date": date,
                "tmax": tmax,
                "tmin": tmin,
                "pluie_pct": rain_pct,
                "chaleur_pct": heat_pct,
            })
        return rows, True

    except Exception:
        month = datetime.now().month
        rainy = month in (6, 7, 8, 9, 10)

        rows = []
        for i in range(7):
            d = datetime.now() + timedelta(days=i)
            tmax = (
                36 + np.random.uniform(-3, 4)
                if not rainy
                else 31 + np.random.uniform(-2, 3)
            )
            rain_pct = np.random.uniform(40, 85) if rainy else np.random.uniform(0, 15)
            heat_pct = float(np.clip((tmax - 32) / 13 * 100, 0, 100))
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "tmax": round(tmax, 1),
                "tmin": round(tmax - 8, 1),
                "pluie_pct": round(rain_pct, 0),
                "chaleur_pct": round(heat_pct, 0),
            })
        return rows, False


# ============================================================================
# INITIALISATION
# ============================================================================

init_state()

# ============================================================================
# EN-TÊTE — STYLE APPLICATION PC
# ============================================================================

ambient_live, humidity_live, live_ok = get_live_weather()

st.markdown("""
<div class="desktop-titlebar">
  <div class="desktop-brand">⚡ IA RÉSEAU PRO</div>
  <div class="desktop-subtitle">Supervision prédictive · Parc de transformateurs SONABEL · Ouagadougou</div>
</div>
""", unsafe_allow_html=True)

# NAVIGATION UNIQUE — chaque entrée correspond à une seule page.
with st.sidebar:
    st.markdown("## ⚡ IA RÉSEAU PRO")
    st.caption("CENTRE DE SUPERVISION ET DE SIMULATION")
    st.divider()
    nav_groups = [
        ("SUPERVISION", [
            ("🏠", "Tableau de bord", "🏠 Tableau de bord"),
            ("📋", "État des transformateurs", "📋 État des transformateurs"),
            ("🗺️", "Carte du parc", "🗺️ Carte du parc"),
        ]),
        ("ANALYSE", [
            ("📈", "Historique des pannes", "📈 Historique des pannes"),
            ("🌦️", "Météo & prévisions", "🌦️ Météo & prévisions"),
        ]),
        ("SIMULATION", [
            ("🖐️", "Mode manuel", "🖐️ Mode manuel"),
            ("🎲", "Simulation de pannes", "🎲 Simulation de pannes"),
        ]),
    ]
    for group_title, items in nav_groups:
        st.markdown(f"### {group_title}")
        for icon, label, target in items:
            active = st.session_state.section == target
            if st.button(
                f"{'●' if active else '○'}  {icon}  {label}",
                key=f"nav_{target}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.section = target
                st.rerun()
    st.divider()
    st.markdown("### CONTRÔLE")
    if st.button(
        "⏸️ Mettre en pause" if st.session_state.running else "▶️ Reprendre",
        key="sidebar_run_toggle",
        use_container_width=True,
    ):
        st.session_state.running = not st.session_state.running
        st.rerun()
    st.caption("Les paramètres avancés du moteur IA restent inchangés.")

# Barre d'outils = actions système uniquement, sans seconde navigation.
st.markdown('<div class="toolbar"><div class="toolbar-title">OUTILS SYSTÈME</div>', unsafe_allow_html=True)
tb1, tb2, tb3, tb4 = st.columns([1.1, 1.1, 1.1, 4.7])
with tb1:
    if st.button("🔄 Actualiser", key="tb_refresh", use_container_width=True):
        st.rerun()
with tb2:
    if st.button(
        "⏸️ Pause" if st.session_state.running else "▶️ Reprendre",
        key="tb_pause",
        use_container_width=True,
    ):
        st.session_state.running = not st.session_state.running
        st.rerun()
with tb3:
    if st.button("📄 Rapport simulation", key="tb_report", use_container_width=True):
        st.session_state.section = "🎲 Simulation de pannes"
        st.rerun()
with tb4:
    source = "🟢 Météo connectée" if live_ok else "🟠 Météo hors ligne — estimation"
    st.markdown(
        f'<div style="text-align:right;padding:9px 8px;font-size:13px;">{source} · '
        f'{ambient_live:.1f} °C · {humidity_live:.0f} % HR · {datetime.now().strftime("%H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

running_label = "EN SERVICE" if st.session_state.running else "EN PAUSE"
model_label = "XGBoost chargé" if MODEL is not None else "Mode heuristique"
s1, s2, s3, s4 = st.columns([1.2, 1.8, 2.0, 2.0])
with s1:
    st.markdown(f"**État :** {running_label}")
with s2:
    st.markdown(f"**Moteur :** {model_label}")
with s3:
    st.markdown(f"**Heure :** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
with s4:
    st.markdown(f"**Parc :** {len(TRANSFORMERS)} transformateurs")

section = st.session_state.section

# Routage exclusif : une seule page de contenu est rendue à chaque exécution.
PAGE_META = {
    "🏠 Tableau de bord": ("🏠 TABLEAU DE BORD", "Vue globale du parc et indicateurs de supervision."),
    "🖐️ Mode manuel": ("🖐️ MODE MANUEL", "Réglage des grandeurs électriques et des conditions ambiantes."),
    "🎲 Simulation de pannes": ("🎲 SIMULATION DE PANNES", "Scénarios de panne, progression, risques et conduite à tenir."),
    "📋 État des transformateurs": ("📋 ÉTAT DES TRANSFORMATEURS", "État courant des trois transformateurs surveillés."),
    "📈 Historique des pannes": ("📈 HISTORIQUE DES PANNES", "Épisodes enregistrés et évolution des risques."),
    "🌦️ Météo & prévisions": ("🌦️ MÉTÉO & PRÉVISIONS", "Conditions actuelles et prévisions utiles à la supervision."),
    "🗺️ Carte du parc": ("🗺️ CARTE DU PARC", "Localisation et niveau de risque des transformateurs."),
}
page_title, page_subtitle = PAGE_META.get(section, PAGE_META[SECTIONS[0]])
st.markdown(
    f'<div class="page-header"><h2>{page_title}</h2><p>{page_subtitle}</p></div>',
    unsafe_allow_html=True,
)


# ============================================================================
# HABILLAGE 2.0 — EN-TÊTE DE SUPERVISION
# ============================================================================

def render_command_header(active):
    """En-tête dynamique commun aux pages de l'application."""
    now = datetime.now().strftime("%d/%m/%Y · %H:%M:%S")
    active_count = sum(1 for t in TRANSFORMERS if st.session_state.last_status[t["id"]]["risk"] >= 50)
    critical_count = sum(1 for t in TRANSFORMERS if st.session_state.last_status[t["id"]]["risk"] >= 65)
    clean = active.replace("🏠 ","").replace("🖐️ ","").replace("🎲 ","").replace("📋 ","").replace("📈 ","").replace("🌦️ ","").replace("🗺️ ","")
    html = f'''
    <div class="tech-panel" style="margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;">
        <div>
          <div class="tech-kicker">CENTRE DE SUPERVISION · RÉSEAU DE DISTRIBUTION</div>
          <div class="tech-value">{clean}</div>
          <div class="tech-muted">IA RÉSEAU PRO 2.0 · moteur de risque et anticipation des signatures de panne</div>
        </div>
        <div style="text-align:right;min-width:250px;">
          <div><span class="live-dot"></span> <b>SUPERVISION ACTIVE</b></div>
          <div class="tech-muted" style="margin-top:5px;">{now} · {len(TRANSFORMERS)} transformateurs</div>
          <div style="margin-top:7px;">
            <span class="tech-chip">⚠ Surveillance : {active_count}</span>
            <span class="tech-chip">🔴 Critiques : {critical_count}</span>
          </div>
        </div>
      </div>
    </div>'''
    st.markdown(html, unsafe_allow_html=True)


def render_fleet_cards():
    """Cartes compactes du parc pour une lecture instantanée."""
    cols = st.columns(len(TRANSFORMERS))
    for col, t in zip(cols, TRANSFORMERS):
        status = st.session_state.last_status[t["id"]]
        risk = float(status.get("risk", 0.0))
        emoji = status.get("emoji", "🟢")
        label = status.get("status", "Faible")
        failure = status.get("failure") or "Aucune panne dominante"
        with col:
            html = f'''
            <div class="fleet-card">
              <div class="fleet-name">{emoji} {t["nom"]} <span style="font-size:11px;color:#6d8499;">· {t["quartier"]}</span></div>
              <div class="fleet-risk">{risk:.0f}<span style="font-size:15px;"> %</span></div>
              <div style="font-size:11px;font-weight:800;color:#46627c;">{label}</div>
              <div class="fleet-line"><div class="fleet-fill" style="width:{max(0,min(100,risk))}%;"></div></div>
              <div style="font-size:11px;color:#61798e;margin-top:8px;">IA : <b>{failure}</b></div>
            </div>'''
            st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# SECTION 0 — TABLEAU DE BORD
# ============================================================================

def render_dashboard():
    """Rendu exclusif de la section 🏠 Tableau de bord."""
    st.subheader("🏠 Vue opérationnelle du parc")
    st.caption(
        "Cette vue synthétise le dernier état enregistré en mode manuel. "
        "La priorité est indicative et doit être calibrée avec les données "
        "réelles, la criticité des départs et les procédures d'exploitation."
    )

    render_fleet_cards()
    st.markdown("### 📡 État instantané du parc")

    rows = []
    for t in TRANSFORMERS:
        status = st.session_state.last_status[t["id"]]
        priority, priority_score = transformer_priority(t, status["risk"])
        rows.append({
            "Transformateur": t["nom"],
            "Quartier": t["quartier"],
            "Risque (%)": round(status["risk"], 1),
            "Statut": f"{status['emoji']} {status['status']}",
            "Tendance": status["trend_cat"],
            "Priorité": priority,
            "Score priorité": round(priority_score, 1),
            "Source": status["source"],
        })

    df_dash = pd.DataFrame(rows)

    k1, k2, k3, k4 = st.columns(4)
    avg_risk = float(df_dash["Risque (%)"].mean()) if len(df_dash) else 0
    high_count = int((df_dash["Risque (%)"] >= 65).sum())
    watch_count = int((df_dash["Risque (%)"] >= 50).sum())

    k1.metric("Transformateurs suivis", len(TRANSFORMERS))
    k2.metric("Risque moyen", f"{avg_risk:.1f} %")
    k3.metric("Risque élevé", high_count)
    k4.metric("À surveiller (≥ 50 %)", watch_count)

    st.dataframe(
        df_dash.sort_values(["Score priorité", "Risque (%)"], ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🚨 Alertes prioritaires")

    alerts = df_dash.sort_values("Score priorité", ascending=False).head(3)

    for _, row in alerts.iterrows():
        msg = (
            f"**{row['Transformateur']} — {row['Priorité']}** | "
            f"Risque {row['Risque (%)']:.1f} % | {row['Tendance']}"
        )
        if row["Score priorité"] >= 65:
            st.error(msg)
        elif row["Score priorité"] >= 45:
            st.warning(msg)
        else:
            st.info(msg)

    st.markdown("### 📈 Tendances du parc")

    history_rows = []
    for t in TRANSFORMERS:
        for point in list(st.session_state.history[t["id"]]):
            history_rows.append({
                "Temps": point["t"],
                "Transformateur": t["nom"],
                "Risque (%)": point["risk"],
                "Température huile (°C)": point.get("oil_temp"),
                "Charge": point.get("charge"),
            })

    if history_rows:
        df_hist = pd.DataFrame(history_rows)
        fig = px.line(
            df_hist,
            x="Temps",
            y="Risque (%)",
            color="Transformateur",
            title="Évolution du risque",
        )
        fig.update_yaxes(range=[0, 100])
        fig.add_hline(y=50, line_dash="dash")
        fig.add_hline(y=65, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True, key="dashboard_risk_chart")

        st.caption(
            "Les courbes représentent les valeurs enregistrées par "
            "l'application, et non une mesure SCADA temps réel."
        )
    else:
        st.info("Lancez le mode manuel pour alimenter l'historique.")

    st.markdown("### 🧠 Explication du risque")

    selected_name = st.selectbox(
        "Transformateur à analyser",
        [t["nom"] for t in TRANSFORMERS],
        key="dashboard_explain_transformer",
    )
    selected = next(t for t in TRANSFORMERS if t["nom"] == selected_name)
    selected_status = st.session_state.last_status[selected["id"]]
    features = selected_status.get("features", {})

    if features:
        explain_df = local_risk_sensitivity(features, selected["type"])
        st.caption(
            f"Source : {explain_df.attrs.get('source', 'analyse locale')}. "
            "Une sensibilité locale n'est pas une causalité."
        )
        st.dataframe(explain_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune mesure récente pour ce transformateur.")

    st.markdown("### 📄 Rapport de simulation")
    st.markdown(
        '<div class="report-box">Le rapport est construit à partir des scénarios sélectionnés dans '
        '<b>Simulation de pannes</b>. Il contient les résultats, la phase de dégradation, les actions '
        'réseau, les directives techniques et un commentaire sur la conduite à tenir.</div>',
        unsafe_allow_html=True,
    )
    if st.button("🎲 Ouvrir la simulation pour établir le rapport", key="dashboard_open_sim"):
        st.session_state.section = "🎲 Simulation de pannes"
        st.rerun()
    st.download_button(
        "⬇️ Exporter le rapport complet de simulation",
        data=build_intervention_report(),
        file_name=f"rapport_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
        key="dashboard_report_download",
    )


# ============================================================================
# SECTION 1 — MODE MANUEL
# ============================================================================

def _compact_risk_color(risk):
    if risk >= 65: return "#ef4656"
    if risk >= 50: return "#f6a623"
    if risk >= 35: return "#d99000"
    return "#19b875"


def _compact_interpretation(t, features, risk, slope, simulated=None):
    rows = anticipated_failures(features, risk)
    top = rows[0] if rows else None
    band, emoji = risk_band(risk)
    ai = ai_decision(features, t["type"], risk, 0.0)
    parts = [f"{emoji} Niveau {band.lower()} : risque global {risk:.1f} %. "]
    if top:
        parts.append(f"Signature dominante : « {top[1]} » ({top[2]:.0f} % du risque global estimé). ")
    if ai.get("driver"):
        parts.append(f"Variable la plus sensible localement : {ai['driver']}. ")
    if simulated:
        parts.append(f"Scénario actif : {FAILURE_LABELS.get(simulated, simulated)}.")
    return "".join(parts)


def render_manual_mode():
    """Poste manuel : toutes les grandeurs affichées suivent instantanément les commandes."""
    st.markdown(
        "<div class='compact-head'><div><div class='compact-head-title'>🖐️ POSTE DE PILOTAGE MANUEL</div>"
        "<div class='compact-head-sub'>Réglage instantané · XGBoost · surveillance des limites de mesure</div>"
        "</div><span class='live-dot'></span></div>",
        unsafe_allow_html=True,
    )

    top1, top2, top3 = st.columns([1.1, 1.0, 1.0])
    with top1:
        tid = st.selectbox(
            "Transformateur", [t["id"] for t in TRANSFORMERS],
            format_func=lambda x: next(t["nom"] for t in TRANSFORMERS if t["id"] == x),
            key="manual_selected_transformer",
        )
    t = next(x for x in TRANSFORMERS if x["id"] == tid)
    i_nom = nominal_current(t)
    with top2:
        season_options = list(SEASON_PRESETS.keys())
        manual_season = st.selectbox(
            "Saison", season_options,
            index=season_options.index(st.session_state.manual_season),
            key="manual_season_select",
        )
        st.session_state.manual_season = manual_season
    with top3:
        st.markdown(
            f"<div class='control-panel' style='margin-top:3px'><div class='control-title'>⚙️ Référence</div>"
            f"<div style='font-size:13px'>Puissance : <b>{RATINGS_KVA[t['nom']]} kVA</b><br>"
            f"Courant nominal : <b>{i_nom:.0f} A</b></div></div>",
            unsafe_allow_html=True,
        )

    # Valeurs de référence saines : elles servent uniquement à initialiser les curseurs.
    base_charge = 0.80
    base_voltage = 1.00
    base_oil = float(np.clip(45 + 28 * base_charge, 35, 120))
    base_hum = float(np.clip(SEASON_PRESETS[manual_season]["humidity"], 5, 100))
    base_amb = 32.0

    left, center, right = st.columns([1.12, 1.0, 1.25], gap="small")
    with left:
        st.markdown("<div class='control-panel'><div class='control-title'>🎛️ Commandes manuelles</div>", unsafe_allow_html=True)
        volts = st.slider("Tension BT (V)", 350.0, 460.0, base_voltage * U_NOM, 2.0, key=f"m_v_{tid}")
        max_amp = float(max(100, round(1.5 * i_nom / 5) * 5))
        amps = st.slider(
            "Courant de charge (A)", 0.0, max_amp,
            float(np.clip(base_charge * i_nom / max(base_voltage, .1), 0, max_amp)), 5.0,
            key=f"m_i_{tid}"
        )
        oil_temp = st.slider("Température huile (°C)", 35.0, 120.0, base_oil, 1.0, key=f"m_oil_{tid}")
        ambient = st.slider("Température ambiante (°C)", 15.0, 50.0, base_amb, .5, key=f"m_amb_{tid}")
        humidity = st.slider("Humidité (%)", 5.0, 100.0, base_hum, 1.0, key=f"m_hum_{tid}")
        imbalance = st.slider("Déséquilibre phases (%)", 0.0, 30.0, 0.0, 1.0, key=f"m_imb_{tid}")
        st.markdown("</div>", unsafe_allow_html=True)

    voltage_pu = volts / U_NOM
    current_ratio = amps / max(i_nom, 1e-6)
    charge = float(np.clip(voltage_pu * current_ratio, 0.0, 1.5))
    # Indicateur visuel dérivé du déséquilibre et de la charge. Ce n'est pas une
    # variable du modèle XGBoost et il ne doit pas être présenté comme un angle
    # de phase mesuré sans capteur dédié.
    dephasage_indicatif = float(np.clip(5.0 + 0.55 * imbalance + 8.0 * max(0.0, charge - 0.8), 5.0, 35.0))

    features = {
        "charge": charge, "oil_temp": oil_temp, "ambient": ambient,
        "humidity": humidity, "voltage": voltage_pu, "season_label": manual_season
    }
    risk = predict_risk(features, t["type"])
    if imbalance > 10:
        risk = float(np.clip(risk + (imbalance - 10) * 0.35, 1.0, 100))
    push_history(tid, risk, features)
    slope, trend_cat = compute_trend(tid)
    band, emoji = risk_band(risk)
    update_last_status(tid, risk, slope, trend_cat, band, emoji, source="manuel", failure=None, progress=0.0, features=features)
    boundary_risk, boundary_alerts = measurement_boundary_risk(features)

    with center:
        color = _compact_risk_color(risk)
        rows = anticipated_failures(features, risk)
        top = rows[0] if rows else None
        panne = top[1] if top else "Aucune"
        pct = top[2] if top else 0
        st.markdown(
            f"<div class='ai-panel'><div class='ai-small'>🤖 RISQUE IA — {t['nom']}</div>"
            f"<div class='ai-big' style='color:{color}'>{risk:.1f}%</div>"
            f"<div class='ai-badge'>{emoji} {band}</div>"
            f"<div style='font-size:10px;color:#8fb4c9;margin-top:8px'>0 % = absence de risque absolue · 100 % = niveau maximal</div>"
            f"</div>", unsafe_allow_html=True
        )
        st.progress(min(1.0, risk / 100.0), text=f"Jauge de risque · 0 % ───────── 100 % · {risk:.1f} %")
        if boundary_alerts:
            st.warning("⚠️ **Contrôle des limites :** " + " ".join(boundary_alerts[:2]))
        else:
            st.caption("✓ Les grandeurs restent dans une zone cohérente avec les limites surveillées.")
        if top:
            st.markdown(f"**Panne à anticiper :** {panne} · part estimée {pct:.1f} %")
        for _, label, p in rows[:3]:
            st.progress(min(1, p / 100), text=f"{label.capitalize()} · {p:.1f}%")

    with right:
        st.markdown("<div class='control-panel'><div class='control-title'>🧠 Interprétation instantanée</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='interpret-box'>{_compact_interpretation(t, features, risk, slope)}</div>", unsafe_allow_html=True)
        ai = ai_decision(features, t["type"], risk)
        if ai.get("driver"):
            st.markdown(f"**Variable dominante :** {ai['driver']} ({ai['effect']:+.1f} pt)")
        if ai.get("action") and risk >= 35:
            st.markdown(f"**Action :** {ai['action']}")
        if boundary_risk > 0:
            st.info(f"**Surveillance capteurs / limites :** contribution {boundary_risk:.1f} pt au risque.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Bandeau inférieur : les valeurs sont calculées à partir des commandes actuelles,
    # donc aucune valeur fixe ne reste affichée lorsque l'utilisateur agit sur les curseurs.
    vals = [
        ("Tension", f"{volts:.0f} V"),
        ("Courant", f"{amps:.0f} A"),
        ("Charge", f"{charge*100:.0f} %"),
        ("Huile", f"{oil_temp:.0f} °C"),
        ("Humidité", f"{humidity:.0f} %"),
        ("Déphasage indicatif", f"{dephasage_indicatif:.1f}°"),
    ]
    st.markdown(
        "<div class='sim-strip'>" +
        "".join(f"<div class='sim-cell'><div class='sim-label'>{a}</div><div class='sim-val'>{b}</div></div>" for a,b in vals) +
        "</div>", unsafe_allow_html=True
    )
    st.caption("Le déphasage indicatif est un indicateur dérivé de l'état de charge et du déséquilibre ; il n'est pas une mesure directe du facteur de puissance.")

    hist = list(st.session_state.history.get(tid, []))
    if len(hist) >= 2:
        dfh = pd.DataFrame(hist).tail(35)
        fig = px.line(dfh, x="t", y="risk", markers=True)
        fig.update_layout(height=180, margin=dict(l=5,r=5,t=8,b=5), yaxis=dict(range=[0,100], title="Risque (%)"), xaxis_title=None, showlegend=False)
        fig.add_hline(y=50, line_dash="dash", line_color="orange")
        fig.add_hline(y=65, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True, key=f"manual_compact_chart_{tid}")
    else:
        st.caption("Déplacez un curseur : la courbe du risque se construit automatiquement ici.")



# ============================================================================
# SECTION 2 — SIMULATION DE PANNES
# ============================================================================

def render_simulation_mode():
    """Laboratoire de panne compact : tout le diagnostic principal tient dans une seule vue."""
    st.markdown("<div class='compact-head'><div><div class='compact-head-title'>🎲 LABORATOIRE DE SIMULATION DE PANNES</div><div class='compact-head-sub'>Scénario → dégradation physique → XGBoost → interprétation</div></div><span class='live-dot'></span></div>", unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns([1.0,1.35,1.0,1.0])
    with c1:
        tid=st.selectbox("Transformateur",[t["id"] for t in TRANSFORMERS],format_func=lambda x:next(t["nom"] for t in TRANSFORMERS if t["id"]==x),key="sim_selected_transformer")
    t=next(x for x in TRANSFORMERS if x["id"]==tid)
    options=["Aucune (fonctionnement normal)"]+list(FAILURE_MODES.keys())
    current=st.session_state.sim_failure.get(tid) or options[0]
    with c2:
        choice=st.selectbox("Scénario de panne",options,index=options.index(current) if current in options else 0,key=f"sim_choice_compact_{tid}")
    with c3:
        sim_temp=st.slider("Température ambiante",15.0,50.0,float(st.session_state.simulation_temperature),.5,key="sim_temp_compact")
        st.session_state.simulation_temperature=sim_temp
    with c4:
        seasons=list(SEASON_PRESETS.keys())
        sim_season=st.selectbox("Saison",seasons,index=seasons.index(st.session_state.simulation_season),key="sim_season_compact")
        st.session_state.simulation_season=sim_season

    previous=st.session_state.sim_failure.get(tid) or options[0]
    if choice!=previous:
        if choice==options[0]:
            st.session_state.sim_failure[tid]=None; st.session_state.sim_start_time[tid]=None
        else:
            st.session_state.sim_failure[tid]=choice; st.session_state.sim_start_time[tid]=time.time()
        st.session_state.simulation_history[tid].clear()
        update_simulation_status(tid,0.0,0.0,"stable","Faible","🟢",source="simulation",failure=None if choice==options[0] else choice,progress=0.0,features={})
        st.rerun()

    active=st.session_state.sim_failure.get(tid)
    climate=TYPE_CLIMATE.get(t["type"],{"ambient_factor":1.0,"humidity_factor":1.0,"orage_voltage_bonus":0.0})
    weather_name=st.selectbox("Météo",list(WEATHER_PRESETS.keys()),key="sim_weather_compact")
    weather=WEATHER_PRESETS[weather_name]
    oil_add=max(0.0,sim_temp-30)*0.6*climate["ambient_factor"]
    hum=float(np.clip(SEASON_PRESETS[sim_season]["humidity"]+weather["humidity_bonus"]*climate["humidity_factor"],5,100))
    vadd=climate["orage_voltage_bonus"] if weather["orage"] else 0.0
    base={"charge":0.8,"oil_temp":45+0.8*28+oil_add,"voltage":1.0+vadd}

    if active:
        start=st.session_state.sim_start_time.get(tid) or time.time(); st.session_state.sim_start_time[tid]=start
        elapsed=max(0.0,time.time()-start)
        tau=FAILURE_TIME_CONSTANT*TYPE_TIME_FACTOR.get(t["type"],1.0)
        progress=1-math.exp(-elapsed/tau)
        effect=FAILURE_MODES[active]["effet"]
        charge=base["charge"]+effect.get("charge",0)*progress
        oil=base["oil_temp"]+effect.get("oil_temp",0)*progress
        voltage=base["voltage"]+effect.get("voltage",0)*progress
        humidity=hum+effect.get("humidity",0)*progress
        ambient=sim_temp+effect.get("ambient",0)*progress
    else:
        progress=0.0; charge=base["charge"]; oil=base["oil_temp"]; voltage=base["voltage"]; humidity=hum; ambient=sim_temp

    features={"charge":charge,"oil_temp":oil,"ambient":ambient,"humidity":humidity,"voltage":voltage,"season_label":sim_season}
    risk=predict_risk(features,t["type"])
    climate_bonus=0.0
    if sim_temp>35: climate_bonus+=(sim_temp-35)*0.7*climate["ambient_factor"]
    if humidity>75: climate_bonus+=(humidity-75)*0.10*climate["humidity_factor"]
    if weather["orage"]: climate_bonus+=5.0*climate["ambient_factor"]
    ref={"charge":base["charge"],"oil_temp":base["oil_temp"],"ambient":sim_temp,"humidity":hum,"voltage":base["voltage"],"season_label":sim_season}
    if SIM_RELATIVE_RISK: climate_bonus-=predict_risk(ref,t["type"])*(1-progress)
    risk=float(np.clip(risk+climate_bonus,0,100)) if active else 0.0

    push_simulation_history(tid,risk,features)
    slope,trend=compute_simulation_trend(tid)
    band,emoji=risk_band(risk)
    update_simulation_status(tid,risk,slope,trend,band,emoji,source="simulation",failure=active,progress=progress,features=features)
    st.session_state.simulation_status[tid]["reference"]=ref

    left,center,right=st.columns([1.05,1.0,1.3],gap="small")
    with left:
        st.markdown("<div class='control-panel'><div class='control-title'>📡 Grandeurs simulées</div>",unsafe_allow_html=True)
        vals=[("Tension",f"{voltage*U_NOM:.0f} V"),("Courant",f"{charge/max(voltage,.5)*nominal_current(t):.0f} A"),("Charge",f"{charge*100:.0f} %"),("Huile",f"{oil:.1f} °C"),("Ambiante",f"{ambient:.1f} °C"),("Humidité",f"{humidity:.0f} %")]
        for a,b in vals: st.markdown(f"**{a}** <span style='float:right;font-weight:850'>{b}</span>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        st.progress(min(1,progress),text=f"Dégradation : {progress*100:.0f}%")
        if active:
            phase_label,phase_kind,phase_text=failure_phase(progress)
            getattr(st,phase_kind)(f"{phase_label} — {phase_text}")
        else: st.success("🟢 Fonctionnement normal — aucune panne active")

    with center:
        rows=anticipated_failures(features,risk) if active else []
        top=rows[0] if rows else None
        color=_compact_risk_color(risk)
        panne=top[1] if top else (FAILURE_LABELS.get(active,active) if active else "Aucune")
        pct=top[2] if top else 0
        st.markdown(f"<div class='ai-panel'><div class='ai-small'>🤖 DIAGNOSTIC IA — {t['nom']}</div><div class='ai-big' style='color:{color}'>{risk:.0f}%</div><div class='ai-badge'>{emoji} {band}</div><div class='ai-line'></div><div class='ai-small'>ANTICIPATION</div><div style='font-size:18px;font-weight:850'>{panne}</div><div style='font-size:12px;color:#a9cce0'>Signature estimée : {pct:.0f}% · tendance {slope:+.2f} pt/min</div></div>",unsafe_allow_html=True)
        for _,label,p in rows[:3]: st.progress(min(1,p/100),text=f"{label.capitalize()} · {p:.0f}%")

    with right:
        st.markdown("<div class='control-panel'><div class='control-title'>🧠 Interprétation + conduite</div>",unsafe_allow_html=True)
        if active:
            st.markdown(f"<div class='interpret-box'>{_compact_interpretation(t,features,risk,slope,active)}</div>",unsafe_allow_html=True)
            ai=ai_decision(features,t["type"],risk,climate_bonus,ref)
            if ai.get("driver"): st.markdown(f"**Facteur dominant :** {ai['driver']} ({ai['effect']:+.1f} pt)")
            if ai.get("failure"): st.markdown(f"**Panne anticipée :** {ai['failure'][1]} ≈ {ai['failure'][2]:.0f}%")
            for action in NETWORK_ACTIONS_BY_FAILURE.get(active,[])[:2]: st.markdown(f"• {action}")
            for directive in FAILURE_MODES[active]["directives"][:1]: st.caption(f"🩺 {directive}")
        else:
            st.markdown("<div class='interpret-box'>Sélectionnez un scénario pour voir la dégradation, la montée du risque et l'interprétation IA en temps réel.</div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    hist=list(st.session_state.simulation_history[tid])
    if len(hist)>=2:
        dfh=pd.DataFrame(hist).tail(45)
        fig=px.line(dfh,x="t",y="risk",markers=True)
        fig.update_layout(height=185,margin=dict(l=5,r=5,t=8,b=5),yaxis=dict(range=[0,100],title="Risque (%)"),xaxis_title=None,showlegend=False)
        fig.add_hline(y=50,line_dash="dash",line_color="orange")
        fig.add_hline(y=65,line_dash="dash",line_color="red")
        st.plotly_chart(fig,use_container_width=True,key=f"sim_compact_chart_{tid}")
    else:
        st.caption("La courbe du risque apparaît dès les premiers rafraîchissements de la simulation.")

    vals=[("Tension",f"{voltage*U_NOM:.0f} V"),("Courant",f"{charge/max(voltage,.5)*nominal_current(t):.0f} A"),("Charge",f"{charge*100:.0f}%"),("Huile",f"{oil:.0f}°C"),("Humidité",f"{humidity:.0f}%"),("Phase",f"{progress*100:.0f}%")]
    st.markdown("<div class='sim-strip'>"+"".join(f"<div class='sim-cell'><div class='sim-label'>{a}</div><div class='sim-val'>{b}</div></div>" for a,b in vals)+"</div>",unsafe_allow_html=True)



# ============================================================================
# SECTION 3 — ÉTAT DES TRANSFORMATEURS
# ============================================================================

def render_transformer_status():
    """Rendu exclusif de la section 📋 État des transformateurs."""
    st.subheader("État courant du parc")

    rows = []
    for t in TRANSFORMERS:
        status = st.session_state.last_status[t["id"]]
        rows.append({
            "N°": t["id"],
            "Nom": t["nom"],
            "Type": t["type"],
            "Quartier": t["quartier"],
            "Risque (%)": round(status["risk"], 1),
            "Statut": f"{status['emoji']} {status['status']}",
            "Source": status["source"],
            "Tendance": status["trend_cat"],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "Le tableau utilise le dernier état enregistré en mode manuel. "
        "La simulation de pannes est isolée et n'apparaît pas ici."
    )


# ============================================================================
# SECTION 4 — HISTORIQUE DES PANNES
# ============================================================================

def render_failure_history():
    """Rendu exclusif de la section 📈 Historique des pannes."""
    st.subheader("Historique des épisodes à risque élevé")

    if not st.session_state.event_log:
        st.info("Aucun épisode à risque élevé enregistré pour l'instant.")
        return

    rows = []
    for ev in reversed(st.session_state.event_log):
        fin = ev["fin"]
        duree = ((fin or datetime.now()) - ev["debut"]).total_seconds() / 60
        rows.append({
            "Transformateur": ev["nom"],
            "Cause": ev["type_panne"],
            "Début": ev["debut"].strftime("%H:%M:%S"),
            "Fin": fin.strftime("%H:%M:%S") if fin else "en cours",
            "Durée (min)": round(duree, 1),
            "Risque max (%)": round(ev["risque_max"], 1),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================================
# SECTION 5 — MÉTÉO
# ============================================================================

def render_weather():
    """Rendu exclusif de la section 🌦️ Météo & prévisions."""
    st.subheader("Météo en direct et prévisions à 7 jours")

    source = "en direct (Open-Meteo)" if live_ok else "estimation saisonnière (hors ligne)"

    c1, c2 = st.columns(2)
    c1.metric(f"Température actuelle — {source}", f"{ambient_live:.1f} °C")
    c2.metric("Humidité relative actuelle", f"{humidity_live:.0f} %")

    rows, forecast_ok = get_weekly_forecast()

    st.caption(
        "Prévisions "
        + ("en direct (Open-Meteo)." if forecast_ok else "estimées hors ligne.")
    )

    df = pd.DataFrame(rows)
    df["Jour"] = pd.to_datetime(df["date"]).dt.strftime("%a %d/%m")

    st.dataframe(
        df[["Jour", "tmax", "tmin", "pluie_pct", "chaleur_pct"]].rename(columns={
            "tmax": "T° max (°C)",
            "tmin": "T° min (°C)",
            "pluie_pct": "Probabilité de pluie (%)",
            "chaleur_pct": "Indice de forte chaleur (%)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        df,
        x="Jour",
        y=["pluie_pct", "chaleur_pct"],
        barmode="group",
        labels={"value": "%", "variable": "Indicateur"},
        title="Probabilité de pluie et indice de forte chaleur — 7 prochains jours",
    )
    st.plotly_chart(fig, use_container_width=True, key="weather_forecast_chart")

    render_ai_forecast(rows)


# ============================================================================
# SECTION 6 — CARTE DU PARC
# ============================================================================

def render_map():
    """Rendu exclusif de la section 🗺️ Carte du parc."""
    st.subheader("Carte du parc — dernier état enregistré")

    st.info(
        "La carte utilise le dernier état enregistré en mode manuel pour "
        "chaque transformateur. La simulation de pannes est isolée et "
        "n'est pas affichée ici."
    )

    rows = []
    for t in TRANSFORMERS:
        status = st.session_state.last_status[t["id"]]
        risk = float(status["risk"])
        priority, priority_score = transformer_priority(t, risk)
        etat = "Mode manuel" if status["source"] == "manuel" else "État initial"

        rows.append({
            "N°": t["id"],
            "Nom": t["nom"],
            "Type": t["type"],
            "Quartier": t["quartier"],
            "Risque (%)": round(risk, 1),
            "Statut": f"{status['emoji']} {status['status']}",
            "État": etat,
            "lat": t["lat"],
            "lon": t["lon"],
            "risk": risk,
            # Colonnes dédiées à l'infobulle (noms simples, sans espaces)
            "tip_nom": f"{t['nom']} ({t['type']}) — {t['quartier']}",
            "tip_risque": f"Risque : {risk:.1f} % — {status['emoji']} {status['status']}",
            "tip_priorite": f"Priorité : {priority} ({priority_score:.1f}/100)",
            "tip_etat": etat,
        })

    df_parc = pd.DataFrame(rows)

    st.dataframe(
        df_parc[["N°", "Nom", "Type", "Quartier", "Risque (%)", "Statut", "État"]],
        use_container_width=True,
        hide_index=True,
    )

    def color_for(risk):
        if risk >= 65:
            return [220, 40, 40, 240]      # rouge
        elif risk >= 50:
            return [255, 140, 0, 240]      # orange
        elif risk >= 35:
            return [245, 200, 40, 240]     # jaune
        return [40, 170, 70, 240]          # vert

    df_parc["color"] = df_parc["risk"].apply(color_for)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_parc,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=35,
        radius_min_pixels=7,
        radius_max_pixels=16,
        stroked=True,
        get_line_color=[30, 30, 30, 230],
        line_width_min_pixels=2,
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=OUAGA_LAT, longitude=OUAGA_LON, zoom=13.5, pitch=0)

    tooltip = {"text": "{tip_nom}\n{tip_risque}\n{tip_priorite}\n{tip_etat}"}

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_provider="carto",
            map_style="road",
        )
    )

    st.markdown("### Légende du risque")

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.success("🟢 **0–34 %** — Faible")
    lc2.warning("🟡 **35–49 %** — Modéré")
    lc3.warning("🟠 **50–64 %** — Interpellation")
    lc4.error("🔴 **65–100 %** — Élevé")

    st.markdown("### Priorité indicative d'intervention")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.error("**P1** — Intervention immédiate")
    pc2.warning("**P2** — Intervention prioritaire")
    pc3.warning("**P3** — Surveillance renforcée")
    pc4.info("**P4** — Surveillance normale")

    st.markdown("### Obstacles et contraintes d'accès")

    for t in TRANSFORMERS:
        with st.expander(f"{t['nom']} — {t['quartier']} ({t['type']})"):
            for obstacle in t["obstacles"]:
                st.markdown(f"- {obstacle}")


# ============================================================================
# ROUTEUR UNIQUE — UNE SEULE PAGE DE CONTENU À LA FOIS
# ============================================================================

SECTION_RENDERERS = {
    "🏠 Tableau de bord": render_dashboard,
    "🖐️ Mode manuel": render_manual_mode,
    "🎲 Simulation de pannes": render_simulation_mode,
    "📋 État des transformateurs": render_transformer_status,
    "📈 Historique des pannes": render_failure_history,
    "🌦️ Météo & prévisions": render_weather,
    "🗺️ Carte du parc": render_map,
}


def render_active_page():
    """Routeur strict : UNE et une seule fonction de page est exécutée."""
    active = st.session_state.get("section", SECTIONS[0])
    renderer = SECTION_RENDERERS.get(active)
    if renderer is None:
        active = SECTIONS[0]
        st.session_state.section = active
        renderer = SECTION_RENDERERS[active]

    st.markdown(
        f'<div class="page-runtime-marker">PAGE ACTIVE : {active}</div>',
        unsafe_allow_html=True,
    )
    render_command_header(active)
    renderer()


# ============================================================================
# ACTUALISATION TEMPS RÉEL — PAR FRAGMENT
# ============================================================================
# Le contenu de la page se rafraîchit toutes les secondes SANS relancer tout
# le script avec time.sleep() + st.rerun(). Ainsi chaque changement de page
# termine normalement son exécution et Streamlit supprime les éléments de
# l'ancienne page (plus d'éléments « fantômes » du mode manuel dans la
# simulation). Nécessite Streamlit >= 1.37.

@st.fragment(run_every=1 if st.session_state.running else None)
def live_page():
    render_active_page()


live_page()

# ============================================================================
# BARRE D'ÉTAT INFÉRIEURE — STYLE LOGICIEL PC
# ============================================================================

st.markdown(
    f"""<div class="statusbar"><span>⚡ IA RÉSEAU PRO v2.0 — SUPERVISION INTELLIGENTE</span><span>● Supervision : {running_label}</span><span>● Modèle : {model_label}</span><span>● Météo : {('Connectée' if live_ok else 'Hors ligne')}</span><span>● Transformateurs : {len(TRANSFORMERS)}</span></div>""",
    unsafe_allow_html=True,
)

# ============================================================================
# MESSAGE GLOBAL
# ============================================================================

st.caption(
    "⚠️ Application de démonstration : les grandeurs électriques du mode "
    "manuel, les pannes du mode simulation et la météo simulée sont générées "
    "par l'application. La météo affichée en direct utilise Open-Meteo lorsque "
    "la connexion est disponible. Les priorités, seuils et extrapolations "
    "doivent être validés avant toute utilisation opérationnelle."
)
