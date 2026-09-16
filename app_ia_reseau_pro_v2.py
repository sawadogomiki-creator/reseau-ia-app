"""
Système intelligent de prédiction de défaillance des transformateurs de distribution
--------------------------------------------------------------------------------------
Version 3 — ajouts par rapport à la v2 :
  - Diagnostic de type de panne (surtension, sous-tension, court-circuit,
    surchauffe ambiante, surchauffe interne, humidité) + directives ciblées
    pour corriger ou empêcher chaque type de panne (section 3.6 / 3.7)
  - Mode "Simulation automatique de panne" : 6 scénarios rejouant l'évolution
    Normal -> Surveillance -> Critique d'un type de défaut précis
  - Étude comparative de 3 modes d'installation (cabine maçonnée, préfabriqué,
    haut de poteau) soumis aux mêmes conditions climatiques mais réagissant
    différemment (section 3.5)
  - Parc numéroté avec localisation et type d'installation affichés partout
  - Historique complet des états (pas seulement des pannes) utilisé pour une
    prévision par tendance (régression simple sur les derniers points)
  - Bloc météo en direct (Open-Meteo, sans clé API) donnant la température et
    la situation actuelles à Ouagadougou, avec repli simulé si hors-ligne

La fonction `compute_risk()` reste une formule pondérée illustrative, à
remplacer par le modèle réellement entraîné (chapitre 2) :

    import joblib
    model = joblib.load("mon_modele.pkl")
    proba = model.predict_proba(X)[:, 1] * 100

Lancer localement :
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import time

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import streamlit.components.v1 as components

try:
    import requests
except ImportError:
    requests = None

st.set_page_config(
    page_title="Prédiction de défaillance des transformateurs — Burkina Faso",
    page_icon="⚡",
    layout="wide",
)

# ============================================================================
# STYLE GLOBAL
# ============================================================================
st.markdown("""
<style>
    .block-container {padding-top: 1.4rem;}
    .kpi-card {
        background:#141C2A;border:1px solid #26314A;border-radius:10px;
        padding:16px 18px;height:100%;
    }
    .kpi-label{font-size:0.78rem;color:#8B97AC;margin-bottom:6px;}
    .kpi-value{font-size:1.6rem;font-weight:700;color:#E9ECF2;}
    .kpi-sub{font-size:0.74rem;color:#8B97AC;margin-top:4px;}
    .ref-badge{
        display:inline-block;background:#1B2436;border:1px solid #26314A;
        color:#E8A23D;font-family:monospace;font-size:0.72rem;
        padding:2px 8px;border-radius:5px;margin-left:6px;
    }
    .type-badge{
        display:inline-block;background:#1B2436;border:1px solid #334063;
        color:#4FA3D1;font-size:0.7rem;padding:2px 9px;border-radius:12px;
        margin-left:6px;font-weight:600;
    }
    .section-note{
        background:#141C2A;border-left:3px solid #E8A23D;border-radius:4px;
        padding:10px 14px;font-size:0.85rem;color:#C4CCDA;margin:10px 0 18px;
    }
    .clock-banner{
        background:#141C2A;border:1px solid #26314A;border-radius:10px;
        padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;
        justify-content:space-between;flex-wrap:wrap;gap:10px;
    }
    .clock-time{font-size:1.3rem;font-weight:700;color:#E9ECF2;}
    .clock-sub{font-size:0.78rem;color:#8B97AC;}
    .weather-card{
        background:linear-gradient(135deg,#141C2A 0%,#1B2740 100%);
        border:1px solid #26314A;border-radius:10px;padding:14px 18px;
        display:flex;align-items:center;gap:16px;
    }
    .weather-temp{font-size:2rem;font-weight:700;color:#E9ECF2;}
    .weather-sub{font-size:0.78rem;color:#8B97AC;}
    .fault-card{
        background:#141C2A;border:1px solid #26314A;border-radius:10px;
        padding:14px 16px;cursor:default;
    }
    .fault-tag{
        display:inline-block;font-family:monospace;font-size:0.68rem;
        letter-spacing:0.04em;padding:2px 9px;border-radius:10px;
        text-transform:uppercase;font-weight:700;
    }
    /* --- Console ingénieur / HMI --- */
    .hmi-frame{
        background:#0D141F;border:1px solid #26314A;border-radius:10px;
        padding:14px;position:relative;
    }
    .hmi-frame::before, .hmi-frame::after{
        content:"";position:absolute;width:14px;height:14px;
        border:2px solid #E8A23D;opacity:0.55;
    }
    .hmi-frame::before{top:6px;left:6px;border-right:none;border-bottom:none;}
    .hmi-frame::after{bottom:6px;right:6px;border-left:none;border-top:none;}
    .hmi-title{
        font-family:'JetBrains Mono','IBM Plex Mono',monospace;font-size:0.72rem;
        letter-spacing:0.06em;color:#8B97AC;text-transform:uppercase;margin-bottom:6px;
    }
    .hmi-panel{
        background:#141C2A;border:1px solid #26314A;border-radius:8px;
        padding:12px 14px;
    }
    .hmi-interpret{
        background:#141C2A;border:1px solid #26314A;border-left:3px solid #4FA3D1;
        border-radius:6px;padding:12px 16px;font-size:0.92rem;color:#DCE2EC;line-height:1.55;
    }
</style>
""", unsafe_allow_html=True)


def _rerun():
    """Compatibilité entre les versions de Streamlit (st.rerun ou l'ancien
    st.experimental_rerun). Un rerun côté serveur préserve st.session_state,
    contrairement à un rechargement de page dans le navigateur."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _step_toward(current, target, max_step):
    """Avance 'current' vers 'target' d'au plus 'max_step' — utilisé pour que
    les curseurs produisent un effet progressif (rampe visible) plutôt qu'un
    saut instantané à la valeur cible."""
    if abs(target - current) <= max_step:
        return target
    return current + max_step * (1 if target > current else -1)


def kpi(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def ref(text):
    st.markdown(f'<span class="ref-badge">{text}</span>', unsafe_allow_html=True)


# ============================================================================
# TYPES D'INSTALLATION — modifient l'exposition aux conditions (section 3.5)
# ============================================================================
# swing : amplitude de variation de la température ambiante ressentie par le
#         transformateur par rapport au climat extérieur (abri = amortit)
# offset_amb : décalage fixe (ex : cabine mal ventilée = plus chaud à l'intérieur)
# hum_mult : sensibilité à l'humidité extérieure (abri = protège de la pluie)
# cooling : efficacité de refroidissement (>1 = meilleur refroidissement naturel,
#           <1 = ventilation limitée -> huile plus chaude pour une même charge)

TRANSFO_TYPES = {
    "Cabine maçonnée": dict(
        swing=0.6, offset_amb=3.0, hum_mult=0.65, cooling=0.82,
        icon="🏚️",
        note="Local fermé en dur : protégé de la pluie directe et des écarts brusques de "
             "température, mais la ventilation naturelle est limitée, ce qui pénalise "
             "l'évacuation de la chaleur de l'huile en cas de forte charge.",
    ),
    "Préfabriqué": dict(
        swing=0.85, offset_amb=1.0, hum_mult=0.85, cooling=1.0,
        icon="🏗️",
        note="Poste préfabriqué ventilé : exposition intermédiaire, comportement le plus "
             "proche des conditions climatiques ambiantes moyennes, sert de référence.",
    ),
    "Haut de poteau": dict(
        swing=1.15, offset_amb=0.0, hum_mult=1.2, cooling=1.18,
        icon="🗼",
        note="Transformateur aérien totalement exposé : plein soleil et pluie directe "
             "(écarts de température et pics d'humidité plus marqués), mais la "
             "circulation d'air libre autour de la cuve améliore le refroidissement naturel.",
    ),
}


def apply_type(ambient0, humidity0, charge, ttype):
    """Applique les modificateurs du type d'installation à des conditions
    climatiques de base communes, puis recalcule l'huile en conséquence."""
    m = TRANSFO_TYPES[ttype]
    ambient = 30 + (ambient0 - 30) * m["swing"] + m["offset_amb"]
    humidity = _clamp(humidity0 * m["hum_mult"], 5, 98)
    oil = 45 + (0.5 * charge) / m["cooling"] + 0.3 * (ambient - 30)
    return ambient, humidity, oil


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================================
# MOTEUR DE RISQUE (à remplacer par le modèle entraîné du chapitre 2)
# ============================================================================

def _clamp01(x):
    return max(0.0, min(1.0, x))


def compute_risk(charge, oil, ambient, humidity, voltage, season="seche"):
    charge_stress = (
        0 if charge <= 60
        else _clamp01((charge - 60) / 40) * 0.6 if charge <= 100
        else 0.6 + _clamp01((charge - 100) / 50) * 0.4
    )
    oil_stress = (
        0 if oil <= 70
        else _clamp01((oil - 70) / 25) * 0.7 if oil <= 95
        else 0.7 + _clamp01((oil - 95) / 25) * 0.3
    )
    amb_stress = (
        0 if ambient <= 35
        else _clamp01((ambient - 35) / 7) * 0.7 if ambient <= 42
        else 0.7 + _clamp01((ambient - 42) / 8) * 0.3
    )
    hum_stress = (
        0 if humidity <= 55
        else _clamp01((humidity - 55) / 25) * 0.7 if humidity <= 80
        else 0.7 + _clamp01((humidity - 80) / 15) * 0.3
    )
    season_mult = 1.3 if season == "pluvieuse" else 1.0
    hum_stress = _clamp01(hum_stress * season_mult)
    v_stress = _clamp01(abs(voltage) / 15)

    w = {"charge": 30, "oil": 28, "ambient": 13, "humidity": 13, "voltage": 16}
    contribs = {
        "Charge": charge_stress * w["charge"],
        "Température huile": oil_stress * w["oil"],
        "Température ambiante": amb_stress * w["ambient"],
        "Humidité": hum_stress * w["humidity"],
        "Écart de tension": v_stress * w["voltage"],
    }
    total = min(100.0, sum(contribs.values()))
    return total, contribs


def classify(score):
    if score < 30:
        return "Normal", "#49B586"
    elif score < 65:
        return "Surveillance renforcée", "#E8A23D"
    return "Critique", "#E0554F"


# ----------------------------------------------------------------------------
# DIAGNOSTIC DU TYPE DE PANNE — identifie la signature dominante et propose
# une directive ciblée pour la corriger ou l'empêcher d'arriver (section 3.6)
# ----------------------------------------------------------------------------

FAULT_DIRECTIVES = {
    "Surtension": dict(
        color="#B36AE0",
        prevent="Faire vérifier le régulateur de tension et les prises du changeur de "
                "régulation ; signaler l'écart au poste source avant qu'il ne s'aggrave.",
        correct="Délester les charges sensibles, isoler temporairement le départ concerné "
                "et faire intervenir une équipe pour recalibrer la régulation de tension.",
        network=["Soulager le réseau : vérifier les postes voisins alimentés par la même ligne source.",
                 "Vérifier les clients connectés sensibles (équipements électroniques) sur ce départ."],
        short="surtension",
    ),
    "Sous-tension / creux de tension": dict(
        color="#4FA3D1",
        prevent="Vérifier l'équilibrage des phases et la section des conducteurs sur le "
                "départ ; anticiper un renforcement si la chute de tension est récurrente.",
        correct="Réduire la charge sur la ligne concernée, contrôler les connexions "
                "(bornes desserrées, cosses oxydées) et rétablir l'équilibrage des phases.",
        network=["Soulager le réseau : délester une partie des charges en bout de ligne.",
                 "Vérifier les clients connectés en aval, souvent les premiers touchés par la chute de tension."],
        short="sous-tension",
    ),
    "Court-circuit / surcharge extrême": dict(
        color="#E0554F",
        prevent="Contrôler la coordination des protections (fusibles, disjoncteurs) et "
                "vérifier périodiquement l'état des connexions pour éviter tout amorçage.",
        correct="Mettre hors tension immédiatement, consigner l'ouvrage, puis inspecter "
                "les enroulements et les protections avant toute remise en service.",
        network=["Soulager le réseau : délester les départs non prioritaires avant toute manœuvre.",
                 "Vérifier les clients connectés en aval avant la remise sous tension."],
        short="court-circuit",
    ),
    "Surchauffe ambiante": dict(
        color="#E8A23D",
        prevent="Améliorer la ventilation du site, protéger le transformateur de "
                "l'ensoleillement direct et limiter la charge aux heures les plus chaudes.",
        correct="Réduire temporairement la charge et forcer la ventilation si possible "
                "jusqu'au retour de la température ambiante à un niveau normal.",
        network=["Soulager le réseau aux heures les plus chaudes en décalant les grosses charges.",
                 "Informer les clients concernés d'une possible coupure préventive de courte durée."],
        short="surchauffe ambiante",
    ),
    "Surchauffe interne (refroidissement déficient)": dict(
        color="#E8734A",
        prevent="Planifier un contrôle régulier des radiateurs, ventilateurs et du niveau "
                "d'huile pour garantir une dissipation thermique correcte.",
        correct="Contrôler et déboucher le système de refroidissement, vérifier le niveau "
                "et la qualité de l'huile, et réduire la charge en attendant l'intervention.",
        network=["Soulager le réseau : réduire la charge sur ce poste en attendant l'intervention.",
                 "Vérifier les clients connectés pour une éventuelle surconsommation anormale récente."],
        short="surchauffe interne",
    ),
    "Humidité / infiltration": dict(
        color="#49A3B5",
        prevent="Contrôler l'étanchéité de la cuve et des joints, et remplacer "
                "préventivement le gel de silice du respirateur avant la saison pluvieuse.",
        correct="Vérifier l'étanchéité, remplacer le gel de silice saturé et envisager un "
                "séchage de l'huile si l'humidité interne est confirmée par analyse.",
        network=["Vérifier les postes voisins pour un problème similaire (même épisode pluvieux).",
                 "Planifier une tournée d'inspection avant la prochaine pluie annoncée."],
        short="infiltration d'humidité",
    ),
    "Surcharge": dict(
        color="#E8A23D",
        prevent="Surveiller la courbe de charge et lisser les pointes de consommation par "
                "un effacement ou une meilleure répartition des usages.",
        correct="Délester une partie de la charge vers un poste voisin ; si la surcharge "
                "est récurrente, étudier un renforcement de puissance.",
        network=["Soulager le réseau en répartissant la charge vers un poste voisin disponible.",
                 "Vérifier les clients connectés récemment sur ce départ (nouveau raccordement)."],
        short="surcharge",
    ),
}


def urgent_prognosis(fault_label, severity):
    """Formule un pronostic opérationnel du type : « Risque de court-circuit à
    venir : 40% dans environ 5h si rien n'est fait. » Estimation illustrative,
    croissante avec la sévérité détectée — à remplacer par une vraie
    probabilité de défaillance une fois le modèle entraîné disponible."""
    short = FAULT_DIRECTIVES.get(fault_label, {}).get("short", fault_label.lower())
    probability = round(35 + severity * 60)
    hours = max(1, round(20 - severity * 15))
    return f"⚠ Risque de {short} à venir : {probability}% dans environ {hours} h si rien n'est fait."


def diagnose_fault(charge, oil, ambient, humidity, voltage, ttype=None):
    """Retourne (label_du_defaut, severite 0-1, description) en identifiant la
    signature dominante plutôt que le simple facteur le plus pondéré."""
    cooling = TRANSFO_TYPES[ttype]["cooling"] if ttype else 1.0
    expected_oil = 45 + (0.5 * charge) / cooling + 0.3 * (ambient - 30)
    oil_gap = oil - expected_oil

    flags = []
    if voltage >= 8:
        flags.append(("Surtension", _clamp01((voltage - 8) / 7)))
    if voltage <= -8:
        flags.append(("Sous-tension / creux de tension", _clamp01((-voltage - 8) / 7)))
    if charge >= 125:
        flags.append(("Court-circuit / surcharge extrême", _clamp01((charge - 125) / 40)))
    if oil_gap >= 12:
        flags.append(("Surchauffe interne (refroidissement déficient)", _clamp01(oil_gap / 30)))
    if ambient >= 40:
        flags.append(("Surchauffe ambiante", _clamp01((ambient - 40) / 8)))
    if humidity >= 78:
        flags.append(("Humidité / infiltration", _clamp01((humidity - 78) / 17)))
    if charge >= 95 and not flags:
        flags.append(("Surcharge", _clamp01((charge - 95) / 40)))

    if not flags:
        return None, 0.0, "Les valeurs transmises restent dans les plages de fonctionnement normal."

    label, sev = max(flags, key=lambda kv: kv[1])
    return label, sev, None


def explain(v_display, contribs, season):
    sorted_c = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
    total = min(100.0, sum(contribs.values()))
    if total < 30:
        return "Les valeurs transmises restent dans les plages de fonctionnement normal ; aucun facteur ne présente de contribution significative au risque."
    top, second = sorted_c[0], sorted_c[1]
    text = f"Le risque provient principalement de **{top[0].lower()}** ({v_display[top[0]]})"
    if second[1] > 3:
        text += f", combinée à **{second[0].lower()}** ({v_display[second[0]]})"
    if season == "pluvieuse":
        text += ", dans un contexte de saison pluvieuse qui accentue l'effet de l'humidité sur l'isolation"
    return text + "."


def estimate_delay(score):
    if score < 30:
        return None, "aucune urgence"
    elif score < 50:
        return "plusieurs semaines si la tendance se maintient", "surveillance"
    elif score < 65:
        return "quelques jours à une semaine", "surveillance renforcée"
    elif score < 85:
        return "quelques heures à 1-2 jours", "intervention rapprochée"
    else:
        return "risque de défaillance imminente (moins de quelques heures)", "intervention immédiate"


def trend_forecast(history_df, transfo_id, current_score, window=12):
    """Prévision par tendance : régression linéaire simple du risque sur les
    derniers points d'historique de ce transformateur, pour estimer dans
    combien d'heures simulées le seuil critique (65) serait atteint si la
    tendance actuelle se maintient. Retombe sur l'estimation statique si
    l'historique est insuffisant ou si la tendance est stable/décroissante."""
    if history_df is None or history_df.empty:
        return None
    sub = history_df[history_df["id"] == transfo_id].tail(window)
    if len(sub) < 4:
        return None
    x = sub["heure_sim"].values.astype(float)
    y = sub["risque"].values.astype(float)
    if x.max() == x.min():
        return None
    slope = np.polyfit(x, y, 1)[0]
    if slope <= 0.05 or current_score >= 65:
        return None
    hours_to_critical = (65 - current_score) / slope
    if hours_to_critical <= 0 or hours_to_critical > 500:
        return None
    return f"≈ {hours_to_critical:.0f} h au rythme actuel (tendance sur l'historique)"


# ============================================================================
# SCÉNARIOS DE SIMULATION AUTOMATIQUE DE PANNE (au moins 5 exigés — ici 6)
# ============================================================================
# Chaque scénario est une fonction (p in [0,1], ambient0, humidity0) -> dict
# de mesures, qui fait évoluer une signature de panne précise de Normal à
# Critique. "p" représente la progression dans le temps simulé.

def _scn_surtension(p, amb0, hum0):
    return dict(charge=60 + 8 * p, oil=58 + 6 * p, ambient=amb0, humidity=hum0, voltage=1 + 16 * p)


def _scn_soustension(p, amb0, hum0):
    return dict(charge=58 + 6 * p, oil=56 + 5 * p, ambient=amb0, humidity=hum0, voltage=-1 - 15 * p)


def _scn_court_circuit(p, amb0, hum0):
    # reste stable puis montée brutale tardive (signature d'un défaut soudain)
    spike = 0 if p < 0.62 else ((p - 0.62) / 0.38) ** 1.3
    charge = 58 + spike * 115
    return dict(charge=charge, oil=55 + charge * 0.42, ambient=amb0, humidity=hum0, voltage=2 + spike * 6)


def _scn_surchauffe_ambiante(p, amb0, hum0):
    ambient = amb0 + 19 * p
    charge = 58 + 10 * p
    oil = 48 + 0.5 * charge + 0.3 * (ambient - 30)
    return dict(charge=charge, oil=oil, ambient=ambient, humidity=hum0 * (1 - 0.3 * p), voltage=2)


def _scn_surchauffe_interne(p, amb0, hum0):
    # la charge et l'ambiante restent quasi stables : signature d'un
    # refroidissement défaillant (radiateur bouché, niveau d'huile bas...)
    return dict(charge=64 + 3 * p, oil=55 + 58 * p, ambient=amb0, humidity=hum0, voltage=2)


def _scn_humidite(p, amb0, hum0):
    humidity = hum0 + (96 - hum0) * p
    return dict(charge=58 + 6 * p, oil=55 + 5 * p, ambient=amb0 - 4 * p, humidity=humidity, voltage=-1 - 3 * p)


FAULT_SCENARIOS = {
    "Surtension progressive": dict(fn=_scn_surtension, icon="⚡", fault="Surtension",
        desc="La tension s'écarte progressivement de la normale (régulation défaillante)."),
    "Sous-tension / creux de tension": dict(fn=_scn_soustension, icon="🔻", fault="Sous-tension / creux de tension",
        desc="Chute progressive de tension, souvent liée à un déséquilibre ou une ligne surchargée en amont."),
    "Court-circuit / surcharge extrême": dict(fn=_scn_court_circuit, icon="💥", fault="Court-circuit / surcharge extrême",
        desc="Fonctionnement normal puis montée brutale et tardive du courant — signature d'un défaut soudain."),
    "Surchauffe ambiante": dict(fn=_scn_surchauffe_ambiante, icon="🌡️", fault="Surchauffe ambiante",
        desc="Vague de chaleur : la température ambiante grimpe et entraîne l'huile avec elle."),
    "Surchauffe interne (refroidissement déficient)": dict(fn=_scn_surchauffe_interne, icon="🔥", fault="Surchauffe interne (refroidissement déficient)",
        desc="La température de l'huile grimpe alors que la charge et l'ambiante restent stables — refroidissement en cause."),
    "Humidité / infiltration (saison pluvieuse)": dict(fn=_scn_humidite, icon="💧", fault="Humidité / infiltration",
        desc="Montée d'humidité en saison pluvieuse — risque d'infiltration et de dégradation de l'isolation."),
}

# ============================================================================
# COMPOSANTS VISUELS RICHES (HTML/SVG embarqué)
# ============================================================================

def pipeline_visual_html():
    return """
    <div style="font-family:'IBM Plex Sans',sans-serif;background:#0B1119;padding:6px 0;">
    <style>
      .pv-wrap{display:flex;align-items:center;gap:18px;flex-wrap:wrap;}
      .pv-diag{flex:0 0 220px;}
      .pv-flow{flex:1;min-width:260px;display:flex;align-items:center;justify-content:space-between;position:relative;padding:0 6px;}
      .pv-flow::before{content:"";position:absolute;top:15px;left:18px;right:18px;height:1px;background:#26314A;z-index:0;}
      .pv-stage{display:flex;flex-direction:column;align-items:center;gap:6px;position:relative;z-index:1;}
      .pv-dot{width:30px;height:30px;border-radius:50%;background:#E8A23D;border:2px solid #E8A23D;
              display:flex;align-items:center;justify-content:center;font-size:14px;color:#0B1119;font-weight:700;
              animation:pv-fade 1.6s ease-in-out infinite; }
      .pv-stage:nth-child(1) .pv-dot{animation-delay:0s;}
      .pv-stage:nth-child(2) .pv-dot{animation-delay:0.25s;}
      .pv-stage:nth-child(3) .pv-dot{animation-delay:0.5s;}
      .pv-stage:nth-child(4) .pv-dot{animation-delay:0.75s;}
      @keyframes pv-fade{0%,100%{opacity:0.55;box-shadow:0 0 0 rgba(232,162,61,0);}50%{opacity:1;box-shadow:0 0 10px rgba(232,162,61,0.6);}}
      .pv-stage span{font-size:11px;color:#8B97AC;text-align:center;max-width:74px;}
      .sensor-dot{fill:#4FA3D1;animation:sd-pulse 2.4s ease-in-out infinite;}
      @keyframes sd-pulse{0%,100%{opacity:0.6;}50%{opacity:1;}}
      .sensor-label{font-family:monospace;font-size:8.5px;fill:#8B97AC;}
    </style>
    <div class="pv-wrap">
      <div class="pv-diag">
        <svg viewBox="0 0 320 150" width="100%">
          <rect x="20" y="20" width="6" height="120" fill="#2A3550"/>
          <rect x="60" y="55" width="90" height="60" rx="6" fill="#1B2436" stroke="#334063" stroke-width="1.5"/>
          <line x1="60" y1="65" x2="50" y2="65" stroke="#334063" stroke-width="3"/>
          <line x1="60" y1="80" x2="50" y2="80" stroke="#334063" stroke-width="3"/>
          <line x1="60" y1="95" x2="50" y2="95" stroke="#334063" stroke-width="3"/>
          <line x1="60" y1="105" x2="50" y2="105" stroke="#334063" stroke-width="3"/>
          <rect x="80" y="35" width="8" height="22" fill="#3A4770"/>
          <rect x="110" y="35" width="8" height="22" fill="#3A4770"/>
          <rect x="175" y="30" width="26" height="16" rx="3" fill="#1B2436" stroke="#334063" stroke-width="1.2"/>
          <line x1="188" y1="30" x2="188" y2="18" stroke="#334063" stroke-width="1.5"/>
          <circle class="sensor-dot" cx="84" cy="32" r="3.6"/>
          <circle class="sensor-dot" cx="114" cy="32" r="3.6"/>
          <circle class="sensor-dot" cx="105" cy="85" r="3.6"/>
          <circle class="sensor-dot" cx="188" cy="18" r="3.6"/>
          <text x="66" y="15" class="sensor-label">Courant / tension</text>
          <text x="95" y="130" class="sensor-label">T° huile</text>
          <text x="148" y="55" class="sensor-label">T° amb. / humidité</text>
        </svg>
      </div>
      <div class="pv-flow">
        <div class="pv-stage"><div class="pv-dot">1</div><span>Capteurs</span></div>
        <div class="pv-stage"><div class="pv-dot">2</div><span>Prétraitement</span></div>
        <div class="pv-stage"><div class="pv-dot">3</div><span>Modèle IA</span></div>
        <div class="pv-stage"><div class="pv-dot">4</div><span>Résultat</span></div>
      </div>
    </div>
    </div>
    """


def architecture_visual_html():
    layers = [
        ("Couche d'acquisition", "Capteurs, compteurs intelligents, API météo"),
        ("Couche de traitement des données", "Nettoyage, normalisation, mise en forme"),
        ("Couche d'intelligence", "Modèle entraîné — indice de risque + diagnostic de panne"),
        ("Couche applicative", "Comparaison aux seuils, directives correctives, alertes"),
        ("Couche de présentation", "Tableau de bord, carte du parc, supervision"),
    ]
    rows = ""
    for i, (title, desc) in enumerate(layers):
        arrow = '<div style="text-align:center;color:#26314A;font-size:16px;margin:2px 0;">↓</div>' if i > 0 else ""
        rows += f"""
        {arrow}
        <div style="background:#141C2A;border:1px solid #26314A;border-left:3px solid #E8A23D;
                    border-radius:7px;padding:10px 16px;">
            <div style="font-size:0.86rem;font-weight:600;color:#E9ECF2;">{title}</div>
            <div style="font-size:0.76rem;color:#8B97AC;margin-top:2px;">{desc}</div>
        </div>
        """
    return f"""<div style="font-family:'IBM Plex Sans',sans-serif;max-width:520px;">{rows}</div>"""


def plotly_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 30, "color": "#E9ECF2"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B97AC", "tickfont": {"color": "#8B97AC"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#141C2A",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#173226"},
                {"range": [30, 65], "color": "#3A2C14"},
                {"range": [65, 100], "color": "#3A1D1B"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=210, margin=dict(l=20, r=20, t=20, b=0),
        font={"color": "#E9ECF2"},
    )
    return fig


def plotly_factors(contribs, v_display):
    items = sorted(contribs.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#49B586" if v < 10 else "#E8A23D" if v < 20 else "#E0554F" for v in values]
    text = [v_display[n] for n in names]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=text, textposition="outside", textfont={"color": "#8B97AC", "size": 11},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=230, margin=dict(l=10, r=40, t=10, b=10),
        xaxis={"visible": False, "range": [0, 40]},
        yaxis={"color": "#E9ECF2", "tickfont": {"size": 12}},
        font={"color": "#E9ECF2"},
    )
    return fig


def sensor_color(value, warn, crit, absolute=False):
    v = abs(value) if absolute else value
    if v >= crit:
        return "#E0554F"
    if v >= warn:
        return "#E8A23D"
    return "#49B586"


def transformer_hmi_svg(charge, oil, ambient, humidity, voltage, status_label, status_color):
    c_charge = sensor_color(charge, 60, 100)
    c_oil = sensor_color(oil, 70, 95)
    c_amb = sensor_color(ambient, 35, 42)
    c_hum = sensor_color(humidity, 55, 80)
    c_volt = sensor_color(voltage, 5, 10, absolute=True)

    def box(x, y, w, h, label, value, color, lx1, ly1, lx2, ly2):
        return f"""
        <line x1="{lx1}" y1="{ly1}" x2="{lx2}" y2="{ly2}" stroke="{color}" stroke-width="1.6" stroke-dasharray="3,3" opacity="0.8"/>
        <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#141C2A" stroke="{color}" stroke-width="2"/>
        <circle cx="{x + 12}" cy="{y + 14}" r="4" fill="{color}"/>
        <text x="{x + 24}" y="{y + 18}" font-family="monospace" font-size="10" fill="#8B97AC" letter-spacing="0.4">{label}</text>
        <text x="{x + w/2}" y="{y + h - 12}" font-family="monospace" font-size="19" font-weight="700"
              fill="{color}" text-anchor="middle">{value}</text>
        """

    return f"""
    <svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
      <defs>
        <pattern id="hmigrid" width="24" height="24" patternUnits="userSpaceOnUse">
          <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#161F30" stroke-width="1"/>
        </pattern>
      </defs>
      <rect width="680" height="400" fill="url(#hmigrid)"/>
      <line x1="270" y1="45" x2="410" y2="45" stroke="{c_charge}" stroke-width="3"/>
      <line x1="300" y1="45" x2="300" y2="80" stroke="{c_charge}" stroke-width="3"/>
      <line x1="380" y1="45" x2="380" y2="80" stroke="{c_volt}" stroke-width="3"/>
      <circle cx="300" cy="80" r="7" fill="none" stroke="{c_charge}" stroke-width="3"/>
      <circle cx="380" cy="80" r="7" fill="none" stroke="{c_volt}" stroke-width="3"/>
      <rect x="255" y="95" width="170" height="130" rx="12" fill="{status_color}" fill-opacity="0.14" stroke="{status_color}" stroke-width="3.5"/>
      <circle cx="305" cy="160" r="26" fill="none" stroke="{status_color}" stroke-width="2.6"/>
      <circle cx="375" cy="160" r="26" fill="none" stroke="{status_color}" stroke-width="2.6"/>
      <line x1="425" y1="115" x2="445" y2="115" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="140" x2="445" y2="140" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="165" x2="445" y2="165" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="190" x2="445" y2="190" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="210" x2="445" y2="210" stroke="{c_oil}" stroke-width="4"/>
      <line x1="340" y1="225" x2="340" y2="255" stroke="#8B97AC" stroke-width="3"/>
      <line x1="322" y1="255" x2="358" y2="255" stroke="#8B97AC" stroke-width="3"/>
      <line x1="327" y1="261" x2="353" y2="261" stroke="#8B97AC" stroke-width="2.2"/>
      <line x1="332" y1="267" x2="348" y2="267" stroke="#8B97AC" stroke-width="1.6"/>
      <text x="340" y="245" font-family="monospace" font-size="10" fill="#8B97AC" text-anchor="middle">TERRE</text>
      {box(20, 30, 160, 60, "COURANT / CHARGE", f"{charge:.0f}%", c_charge, 180, 60, 300, 80)}
      {box(500, 30, 160, 60, "ÉCART DE TENSION", f"{voltage:+.1f}%", c_volt, 500, 60, 380, 80)}
      {box(500, 130, 160, 60, "T° HUILE", f"{oil:.0f}°C", c_oil, 500, 160, 447, 155)}
      {box(20, 130, 160, 60, "T° AMBIANTE", f"{ambient:.0f}°C", c_amb, 180, 160, 255, 160)}
      {box(20, 260, 160, 60, "HUMIDITÉ RELATIVE", f"{humidity:.0f}%", c_hum, 130, 260, 200, 226)}
      <rect x="240" y="300" width="200" height="66" rx="8" fill="{status_color}" fill-opacity="0.18" stroke="{status_color}" stroke-width="2.4"/>
      <text x="340" y="325" font-family="monospace" font-size="10" fill="#8B97AC" text-anchor="middle" letter-spacing="0.5">STATUT GLOBAL</text>
      <text x="340" y="352" font-family="monospace" font-size="16" font-weight="700" fill="{status_color}" text-anchor="middle">{status_label.upper()}</text>
    </svg>
    """


def transformer_icon_data_uri(color):
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
      <line x1="24" y1="4" x2="24" y2="15" stroke="{color}" stroke-width="3"/>
      <line x1="40" y1="4" x2="40" y2="15" stroke="{color}" stroke-width="3"/>
      <line x1="16" y1="4" x2="32" y2="4" stroke="{color}" stroke-width="3"/>
      <circle cx="24" cy="15" r="3.2" fill="none" stroke="{color}" stroke-width="2.4"/>
      <circle cx="40" cy="15" r="3.2" fill="none" stroke="{color}" stroke-width="2.4"/>
      <rect x="12" y="19" width="40" height="28" rx="5"
            fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="3"/>
      <circle cx="24" cy="33" r="6.5" fill="none" stroke="{color}" stroke-width="2.4"/>
      <circle cx="40" cy="33" r="6.5" fill="none" stroke="{color}" stroke-width="2.4"/>
      <line x1="8" y1="26" x2="12" y2="26" stroke="{color}" stroke-width="2.4"/>
      <line x1="8" y1="34" x2="12" y2="34" stroke="{color}" stroke-width="2.4"/>
      <line x1="8" y1="42" x2="12" y2="42" stroke="{color}" stroke-width="2.4"/>
      <line x1="32" y1="47" x2="32" y2="55" stroke="{color}" stroke-width="3"/>
      <line x1="23" y1="55" x2="41" y2="55" stroke="{color}" stroke-width="3"/>
      <line x1="26" y1="59" x2="38" y2="59" stroke="{color}" stroke-width="2.2"/>
      <line x1="29" y1="62" x2="35" y2="62" stroke="{color}" stroke-width="1.6"/>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def emoji_icon_data_uri(emoji, bg="#1B2436", border="#8B97AC"):
    """Icône ronde générique (obstacles : arbre, bâtiment, barrage...) encodée
    en SVG data-URI, réutilisable par pydeck comme pour les transformateurs."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">
      <circle cx="24" cy="24" r="19" fill="{bg}" stroke="{border}" stroke-width="2"/>
      <text x="24" y="31" font-size="20" text-anchor="middle">{emoji}</text>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


_ICON_CACHE = {
    "#E0554F": transformer_icon_data_uri("#E0554F"),
    "#E8A23D": transformer_icon_data_uri("#E8A23D"),
    "#49B586": transformer_icon_data_uri("#49B586"),
}

# ============================================================================
# MÉTÉO EN DIRECT (Open-Meteo — sans clé API) avec repli simulé hors-ligne
# ============================================================================

WMO_CODES = {
    0: ("Ciel dégagé", "☀️"), 1: ("Plutôt dégagé", "🌤️"), 2: ("Partiellement nuageux", "⛅"),
    3: ("Couvert", "☁️"), 45: ("Brume", "🌫️"), 48: ("Brouillard givrant", "🌫️"),
    51: ("Bruine légère", "🌦️"), 53: ("Bruine", "🌦️"), 55: ("Bruine dense", "🌦️"),
    61: ("Pluie faible", "🌧️"), 63: ("Pluie", "🌧️"), 65: ("Pluie forte", "🌧️"),
    80: ("Averses", "🌦️"), 81: ("Averses fortes", "🌧️"), 82: ("Averses violentes", "⛈️"),
    95: ("Orage", "⛈️"), 96: ("Orage avec grêle", "⛈️"), 99: ("Orage violent avec grêle", "⛈️"),
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_weather(lat, lon):
    """Interroge Open-Meteo pour la météo actuelle. Retourne un dict avec une
    clé 'source' = 'live' ou 'simulee' (repli si hors-ligne / erreur réseau)."""
    if requests is not None:
        try:
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
                "&timezone=Africa%2FOuagadougou"
            )
            r = requests.get(url, timeout=6)
            r.raise_for_status()
            data = r.json()["current"]
            desc, icon = WMO_CODES.get(int(data.get("weather_code", 0)), ("Conditions variables", "🌡️"))
            return dict(
                source="live",
                temperature=data["temperature_2m"],
                humidity=data["relative_humidity_2m"],
                wind=data.get("wind_speed_10m"),
                desc=desc, icon=icon,
            )
        except Exception:
            pass
    # Repli : estimation simulée à partir de l'heure locale (cycle jour/nuit)
    hour = time.localtime().tm_hour
    temp = 30 + 10 * np.sin((hour - 9) / 24 * 2 * np.pi) + 5
    return dict(source="simulee", temperature=round(float(temp), 1), humidity=28,
                wind=None, desc="Estimation hors-ligne", icon="🌡️")


def weather_card(w):
    src_note = "données en direct — Open-Meteo" if w["source"] == "live" else "connexion indisponible — estimation locale"
    wind_txt = f' · vent {w["wind"]:.0f} km/h' if w.get("wind") is not None else ""
    st.markdown(f"""
    <div class="weather-card">
        <div style="font-size:2.4rem;">{w['icon']}</div>
        <div>
            <div class="weather-temp">{w['temperature']:.1f}°C</div>
            <div class="weather-sub">{w['desc']} · humidité {w['humidity']:.0f}%{wind_txt}</div>
            <div class="weather-sub" style="opacity:0.7;">Ouagadougou — {src_note}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ÉTAT DU PARC — 3 transformateurs (un par type d'installation), numérotés,
# localisés dans 3 quartiers réels de Ouagadougou, avec les obstacles
# environnants utiles pour préparer une intervention (section 3.9)
# ============================================================================

FLEET_INFO = [
    dict(id="TR-01", type="Cabine maçonnée", lieu="Gounghin — centre-ville dense",
         lat=12.3672, lon=-1.5288,
         obstacle_icon="🏢", obstacle_label="Bâtiments mitoyens",
         obstacle_detail="Ruelle étroite, accès camion-nacelle limité : prévoir un véhicule léger pour l'intervention.",
         obs_lat=12.3675, obs_lon=-1.5279),
    dict(id="TR-02", type="Préfabriqué", lieu="Ouaga 2000 — zone périurbaine",
         lat=12.3315, lon=-1.4791,
         obstacle_icon="🌳", obstacle_label="Grand arbre (manguier)",
         obstacle_detail="Branches proches de la cuve : élagage recommandé avant toute intervention avec nacelle.",
         obs_lat=12.3318, obs_lon=-1.4785),
    dict(id="TR-03", type="Haut de poteau", lieu="Tanghin — proximité du barrage n°3",
         lat=12.4051, lon=-1.5140,
         obstacle_icon="🌊", obstacle_label="Barrage n°3 (plan d'eau)",
         obstacle_detail="Humidité ambiante localement plus élevée ; accès parfois boueux en saison des pluies.",
         obs_lat=12.4044, obs_lon=-1.5152),
]
FLEET_SIZE = len(FLEET_INFO)

if "sim_hour" not in st.session_state:
    st.session_state.sim_hour = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "historique" not in st.session_state:
    st.session_state.historique = []          # pannes critiques uniquement
if "state_history" not in st.session_state:
    st.session_state.state_history = []       # tous les états, pour la tendance
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()
if "fleet_base" not in st.session_state:
    st.session_state.fleet_base = pd.DataFrame({
        "id": [f["id"] for f in FLEET_INFO],
        "type": [f["type"] for f in FLEET_INFO],
        "lieu": [f["lieu"] for f in FLEET_INFO],
        "lat": [f["lat"] for f in FLEET_INFO],
        "lon": [f["lon"] for f in FLEET_INFO],
        "base_charge": np.random.default_rng(42).uniform(45, 75, FLEET_SIZE),
        "fault_prone": np.random.default_rng(43).uniform(0, 1, FLEET_SIZE) < 0.3,
    })


def fleet_state_at(hour):
    base = st.session_state.fleet_base
    n = len(base)
    day_frac = hour % 24
    season = "pluvieuse" if (hour // 24) % 5 == 4 else "seche"

    ambient0 = 30 + 10 * np.sin((day_frac - 9) / 24 * 2 * np.pi) + (5 if season == "seche" else -3)
    humidity0 = 25 if season == "seche" else 78

    rng_h = np.random.default_rng(hour)
    charge = base["base_charge"].values + 25 * max(0.0, np.sin((day_frac - 7) / 24 * 2 * np.pi))
    overload = (rng_h.uniform(0, 1, n) < 0.04) & base["fault_prone"].values
    charge = charge + overload * rng_h.uniform(40, 85, n)
    voltage = rng_h.normal(0, 4, n)

    ambients, humidities, oils = [], [], []
    for i, ttype in enumerate(base["type"].values):
        amb, hum, oil = apply_type(ambient0, humidity0 + rng_h.normal(0, 4), charge[i], ttype)
        ambients.append(amb); humidities.append(hum); oils.append(oil)

    df = base.copy()
    df["charge"] = np.round(charge, 1)
    df["oil"] = np.round(oils, 1)
    df["ambient"] = np.round(ambients, 1)
    df["humidity"] = np.round(np.clip(humidities, 5, 98), 1)
    df["voltage"] = np.round(voltage, 1)
    df["season"] = season

    scores = [
        compute_risk(r.charge, r.oil, r.ambient, r.humidity, r.voltage, r.season)[0]
        for r in df.itertuples()
    ]
    df["risque"] = np.round(scores, 1)
    df["statut"] = df["risque"].apply(lambda s: classify(s)[0])
    return df


def advance_time(n_steps=1):
    for _ in range(n_steps):
        st.session_state.sim_hour += 1
        df = fleet_state_at(st.session_state.sim_hour)
        for r in df.itertuples():
            st.session_state.state_history.append({
                "heure_sim": st.session_state.sim_hour, "id": r.id, "risque": r.risque,
            })
            if r.statut == "Critique":
                st.session_state.historique.append({
                    "heure_sim": st.session_state.sim_hour, "id": r.id, "risque": r.risque,
                    "charge": r.charge, "oil": r.oil,
                })
    if len(st.session_state.historique) > 800:
        st.session_state.historique = st.session_state.historique[-800:]
    if len(st.session_state.state_history) > 4000:
        st.session_state.state_history = st.session_state.state_history[-4000:]


# ============================================================================
# NAVIGATION
# ============================================================================

st.sidebar.title("⚡ Système de prédiction")
st.sidebar.caption("Transformateurs de distribution — Burkina Faso")
page = st.sidebar.radio("Navigation", [
    "🏠 Vue d'ensemble",
    "🎛️ Partie 1 — Mode manuel (curseurs)",
    "🎲 Partie 2 — Mode simulation aléatoire",
    "🏗️ Comparaison des types d'installation",
    "🗺️ Carte, historique & prévision du parc",
    "📄 Données réelles (CSV)",
])
st.sidebar.markdown("---")
w_sidebar = fetch_live_weather(12.3714, -1.5197)
st.sidebar.markdown(f"**{w_sidebar['icon']} {w_sidebar['temperature']:.1f}°C** — {w_sidebar['desc']}")
st.sidebar.caption(f"Humidité {w_sidebar['humidity']:.0f}% · Ouagadougou")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Le moteur de risque utilisé dans ce prototype est une formule pondérée illustrative. "
    "Remplacez `compute_risk()` par le modèle réellement entraîné (chapitre 2) avant toute "
    "présentation comme résultat final."
)

# ============================================================================
# PAGE — VUE D'ENSEMBLE
# ============================================================================
if page == "🏠 Vue d'ensemble":
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:2px;">
        <div style="font-size:2.1rem;">⚡</div>
        <div>
            <div style="font-size:1.7rem;font-weight:700;color:#E9ECF2;line-height:1.2;">
                Système intelligent de prédiction de défaillance
            </div>
            <div style="font-size:0.88rem;color:#8B97AC;margin-top:2px;">
                Supervision de 3 transformateurs de distribution — Ouagadougou
                <span class="ref-badge">Chapitre 3</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    col_w, col_k = st.columns([1, 2.4])
    with col_w:
        st.markdown("##### Météo en direct")
        weather_card(fetch_live_weather(12.3714, -1.5197))
    with col_k:
        st.markdown("##### Indicateurs clés")
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Transformateurs surveillés", str(FLEET_SIZE), "3 types d'installation")
        with c2: kpi("Points de mesure", "5", "par transformateur, temps réel")
        with c3: kpi("Types de panne détectés", "7", "voir Partie 1 — Mode manuel")
        with c4: kpi("Horloge de simulation", f"h+{st.session_state.sim_hour}", "voir Carte du parc")

    st.write("")
    st.markdown("##### Suivi en direct du parc")
    ref("Horloge en temps réel — risques et pourcentages notifiés chaque seconde")
    live_c1, live_c2 = st.columns([1, 3])
    with live_c1:
        live_on = st.checkbox("⏱️ Activer le suivi en direct", value=st.session_state.get("live_on", True), key="live_on")
    with live_c2:
        st.caption("Le compteur avance chaque seconde ; l'indice de risque de chaque transformateur est réévalué et affiché en direct (l'horloge de simulation avance d'une heure toutes les 4 secondes de suivi).")

    if "live_seconds" not in st.session_state:
        st.session_state.live_seconds = 0

    df_live = fleet_state_at(st.session_state.sim_hour)
    st.markdown(f"""
    <div class="clock-banner">
        <div class="clock-time">⏱ {st.session_state.live_seconds} s écoulées</div>
        <div class="clock-sub">Horloge de simulation : h+{st.session_state.sim_hour} · {'🟢 suivi actif' if live_on else '⏸️ en pause'}</div>
    </div>
    """, unsafe_allow_html=True)

    live_cols = st.columns(FLEET_SIZE)
    for i, r in enumerate(df_live.itertuples()):
        stlabel, stcolor = classify(r.risque)
        icon = TRANSFO_TYPES[r.type]["icon"]
        with live_cols[i]:
            st.markdown(f"""
            <div class="hmi-panel" style="text-align:center;border-color:{stcolor}55;">
                <div style="font-family:monospace;font-size:0.78rem;color:#8B97AC;">{r.id} · {r.lieu}</div>
                <div style="font-size:1.5rem;margin:4px 0;">{icon}</div>
                <div style="font-size:0.66rem;color:#8B97AC;">{r.type}</div>
                <div style="font-family:monospace;font-weight:700;font-size:1.3rem;color:{stcolor};">{r.risque:.0f}%</div>
                <div style="font-size:0.68rem;color:{stcolor};">{stlabel}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("Section 3.9 — chaque notification recalcule l'indice à partir des mêmes capteurs simulés (charge, huile, ambiante, humidité, tension).")

    if live_on:
        time.sleep(1)
        st.session_state.live_seconds += 1
        if st.session_state.live_seconds % 4 == 0:
            advance_time(1)
        _rerun()

    st.write("")
    st.markdown("##### Pipeline de traitement")
    ref("Section 3.4")
    st.markdown('<div class="hmi-frame">', unsafe_allow_html=True)
    components.html(pipeline_visual_html(), height=210, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-note">
    Chaque mesure capteur traverse le prétraitement, le modèle d'IA, puis la couche
    applicative qui calcule l'indice de risque, identifie le <b>type de panne probable</b>
    et propose une <b>directive corrective ciblée</b> (voir « Partie 1 — Mode manuel » et
    « Partie 2 — Mode simulation aléatoire »).
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Comment naviguer ce tableau de bord")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**🎛️ Partie 1 — Mode manuel**")
        st.caption("Réglez charge, courant et tension des 3 transformateurs ; la météo commune montre l'effet de leur environnement, avec une évolution progressive à chaque changement.")
    with d2:
        st.markdown("**🎲 Partie 2 — Mode aléatoire**")
        st.caption("Choisissez un type de panne : le système simule seul son évolution progressive jusqu'au niveau critique.")
    with d3:
        st.markdown("**🗺️ Carte du parc**")
        st.caption("3 emplacements réels à Ouagadougou avec les obstacles alentour, pour préparer les interventions.")

# ============================================================================
# PAGE — PARTIE 1 : MODE MANUEL (3 transformateurs, météo commune, curseurs
# progressifs — chaque changement de curseur s'applique en douceur pour que
# l'on voie l'impact réel se manifester, pas un saut instantané)
# ============================================================================
elif page == "🎛️ Partie 1 — Mode manuel (curseurs)":
    st.title("Partie 1 — Mode manuel : 3 transformateurs")
    ref("Sections 3.4 – 3.6")
    st.caption("La météo (température ambiante et humidité) est commune aux 3 transformateurs — faites-la varier pour voir comment chacun réagit selon son environnement. Charge, courant et tension se règlent individuellement pour chaque transformateur, et chaque changement se répercute progressivement (pas de saut instantané) pour bien voir l'impact réel.")

    st.markdown("##### Météo commune")
    mc1, mc2, mc3, mc4 = st.columns([1, 1, 1, 1])
    with mc1:
        use_live_console = st.checkbox("Utiliser la météo en direct", value=True, key="console_live")
    with mc4:
        season = st.radio("Saison", ["seche", "pluvieuse"],
                           format_func=lambda s: "Sèche" if s == "seche" else "Pluvieuse", horizontal=True)
    if use_live_console:
        w_c = fetch_live_weather(12.3714, -1.5197)
        amb_target, hum_target = w_c["temperature"], w_c["humidity"]
        with mc2: st.metric("T° ambiante commune", f"{amb_target:.1f} °C")
        with mc3: st.metric("Humidité commune", f"{hum_target:.0f} %")
    else:
        with mc2: amb_target = st.slider("T° ambiante commune (°C)", 15, 48, 34)
        with mc3: hum_target = st.slider("Humidité commune (%)", 10, 98, 30 if season == "seche" else 80)

    # Application progressive : les valeurs affichées rattrapent doucement les
    # valeurs cibles (curseurs) à chaque relecture, au lieu de sauter dessus —
    # c'est ce qui rend l'impact d'un changement visible en train de se produire.
    animating = False
    amb_common = _step_toward(st.session_state.get("disp_amb_common", amb_target), amb_target, 1.2)
    hum_common = _step_toward(st.session_state.get("disp_hum_common", hum_target), hum_target, 3.0)
    st.session_state["disp_amb_common"] = amb_common
    st.session_state["disp_hum_common"] = hum_common
    if amb_common != amb_target or hum_common != hum_target:
        animating = True

    st.write("")
    cols3 = st.columns(3)
    for i, (ttype, meta) in enumerate(TRANSFO_TYPES.items()):
        transfo_id = FLEET_INFO[i]["id"]
        with cols3[i]:
            st.markdown(f"**{meta['icon']} {transfo_id} — {ttype}**")
            st.caption(FLEET_INFO[i]["lieu"])
            charge_target = st.slider("Charge / courant (%)", 0, 160, 60, key=f"charge_{i}")
            voltage_target = st.slider("Écart de tension (%)", -15, 15, 2, key=f"voltage_{i}")

            charge_i = _step_toward(st.session_state.get(f"disp_charge_{i}", charge_target), charge_target, 4.0)
            voltage_i = _step_toward(st.session_state.get(f"disp_voltage_{i}", voltage_target), voltage_target, 1.2)
            st.session_state[f"disp_charge_{i}"] = charge_i
            st.session_state[f"disp_voltage_{i}"] = voltage_i
            if charge_i != charge_target or voltage_i != voltage_target:
                animating = True

            ambient_i, humidity_i, oil_i = apply_type(amb_common, hum_common, charge_i, ttype)
            score_i, contribs_i = compute_risk(charge_i, oil_i, ambient_i, humidity_i, voltage_i, season)
            label_i, color_i = classify(score_i)
            fault_i, sev_i, normal_txt_i = diagnose_fault(charge_i, oil_i, ambient_i, humidity_i, voltage_i, ttype)

            st.markdown(f"""
            <div class="hmi-panel" style="border-color:{color_i}55;">
                <table style="width:100%;font-size:0.78rem;color:#DCE2EC;">
                    <tr><td style="color:#8B97AC;">Charge / courant actuels</td><td style="text-align:right;">{charge_i:.0f} %</td></tr>
                    <tr><td style="color:#8B97AC;">Tension actuelle</td><td style="text-align:right;">{voltage_i:+.1f} %</td></tr>
                    <tr><td style="color:#8B97AC;">T° ambiante réelle</td><td style="text-align:right;">{ambient_i:.1f} °C</td></tr>
                    <tr><td style="color:#8B97AC;">T° huile</td><td style="text-align:right;">{oil_i:.1f} °C</td></tr>
                    <tr><td style="color:#8B97AC;">Humidité réelle</td><td style="text-align:right;">{humidity_i:.0f} %</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(plotly_gauge(score_i, color_i), use_container_width=True, config={"displayModeBar": False}, key=f"gauge_{i}")
            st.markdown(f"""
            <div style="padding:8px 14px;border-radius:8px;background:{color_i}22;color:{color_i};
                        font-weight:700;text-align:center;margin-bottom:8px;">{label_i} — {score_i:.0f}/100</div>
            """, unsafe_allow_html=True)

            if fault_i:
                fd_i = FAULT_DIRECTIVES[fault_i]
                st.markdown(f"""
                <div class="fault-card" style="border-left:3px solid {fd_i['color']};">
                    <span class="fault-tag" style="background:{fd_i['color']}22;color:{fd_i['color']};">⚠ {fault_i}</span>
                    <div style="color:{fd_i['color']};font-size:0.82rem;font-weight:600;margin-top:6px;">{urgent_prognosis(fault_i, sev_i)}</div>
                    <div style="margin-top:8px;font-size:0.68rem;color:#8B97AC;text-transform:uppercase;">🛡️ Prévenir</div>
                    <div style="font-size:0.82rem;color:#DCE2EC;margin:2px 0 6px;">{fd_i['prevent']}</div>
                    <div style="font-size:0.68rem;color:#8B97AC;text-transform:uppercase;">🔧 Corriger</div>
                    <div style="font-size:0.82rem;color:#DCE2EC;margin:2px 0 6px;">{fd_i['correct']}</div>
                    <div style="font-size:0.68rem;color:#8B97AC;text-transform:uppercase;">🔌 Actions réseau</div>
                    <div style="font-size:0.82rem;color:#DCE2EC;margin-top:2px;">• {fd_i['network'][0]}<br>• {fd_i['network'][1]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="fault-card" style="border-left:3px solid #49B586;">
                    <span class="fault-tag" style="background:#49B58622;color:#49B586;">✓ Normal</span>
                    <div style="font-size:0.8rem;color:#DCE2EC;margin-top:6px;">{normal_txt_i}</div>
                </div>
                """, unsafe_allow_html=True)
            st.caption(meta["note"])

    st.write("")
    st.markdown("""
    <div class="section-note">
    À météo commune identique, chaque transformateur ressent une température et une humidité
    différentes selon son environnement (cabine maçonnée = amortie mais mal ventilée, haut de
    poteau = pleinement exposé mais mieux refroidi par l'air libre) — c'est ce qui explique que
    leurs indices de risque divergent même sous la même charge et la même tension.
    </div>
    """, unsafe_allow_html=True)

    if animating:
        time.sleep(0.15)
        _rerun()

# ============================================================================
# PAGE — PARTIE 2 : MODE SIMULATION ALÉATOIRE (le système choisit et fait
# évoluer seul le comportement de la panne choisie — 6 scénarios disponibles)
# ============================================================================
elif page == "🎲 Partie 2 — Mode simulation aléatoire":
    st.title("Partie 2 — Mode simulation aléatoire")
    ref("Section 3.6 — 6 scénarios disponibles")
    st.caption("Choisissez un type de défaut : le système simule seul son évolution progressive de Normal à Critique, avec le pronostic et les actions réseau qui apparaissent dès que la panne est identifiée.")

    if "fault_t" not in st.session_state:
        st.session_state.fault_t = 0
    if "fault_running" not in st.session_state:
        st.session_state.fault_running = False
    if "fault_last_tick" not in st.session_state:
        st.session_state.fault_last_tick = time.time()

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.6, 0.9, 0.9, 1.3])
    with ctrl1:
        scenario_name = st.selectbox("Scénario de panne", list(FAULT_SCENARIOS.keys()))
    with ctrl2:
        label_btn = "⏸️ Pause" if st.session_state.fault_running else "▶️ Lecture"
        if st.button(label_btn, use_container_width=True, key="fault_play"):
            st.session_state.fault_running = not st.session_state.fault_running
            st.session_state.fault_last_tick = time.time()
    with ctrl3:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="fault_reset"):
            st.session_state.fault_t = 0
            st.session_state.fault_running = False
    with ctrl4:
        speed = st.slider("Vitesse", 1, 10, 4, key="fault_speed")

    scn = FAULT_SCENARIOS[scenario_name]
    st.markdown(f"""
    <div class="section-note">{scn['icon']} <b>{scenario_name}</b> — {scn['desc']}</div>
    """, unsafe_allow_html=True)

    DURATION = 40  # pas de simulation
    if st.session_state.fault_running:
        if st.session_state.fault_t < DURATION:
            time.sleep(max(0.12, 1.1 - speed / 11))
            st.session_state.fault_t += 1
            _rerun()
        else:
            st.session_state.fault_running = False

    p = st.session_state.fault_t / DURATION
    amb_base = 34  # ambiante de référence pour ces scénarios pédagogiques
    hum_base = 30
    vals = scn["fn"](p, amb_base, hum_base)
    # petite variation aléatoire à chaque étape pour un comportement moins
    # parfaitement lisse — cohérente d'une relecture à l'autre (seed = étape)
    _noise_rng = np.random.default_rng(hash((scenario_name, st.session_state.fault_t)) % (2**31))
    vals = dict(
        charge=max(0, vals["charge"] + _noise_rng.normal(0, 1.5)),
        oil=max(20, vals["oil"] + _noise_rng.normal(0, 1.0)),
        ambient=vals["ambient"] + _noise_rng.normal(0, 0.6),
        humidity=min(98, max(5, vals["humidity"] + _noise_rng.normal(0, 1.5))),
        voltage=vals["voltage"] + _noise_rng.normal(0, 0.4),
    )
    score, contribs = compute_risk(vals["charge"], vals["oil"], vals["ambient"], vals["humidity"], vals["voltage"])
    label, color = classify(score)
    fault_label, fault_sev, _ = diagnose_fault(vals["charge"], vals["oil"], vals["ambient"], vals["humidity"], vals["voltage"])

    st.write("")
    prog1, prog2 = st.columns([3, 1])
    with prog1:
        st.progress(p, text=f"Progression du scénario : étape {st.session_state.fault_t}/{DURATION}")
    with prog2:
        st.markdown(f'<div style="text-align:center;font-family:monospace;color:{color};font-weight:700;">{label}</div>', unsafe_allow_html=True)

    col_hmi, col_diag = st.columns([1.3, 1])
    with col_hmi:
        st.markdown('<div class="hmi-frame">', unsafe_allow_html=True)
        components.html(transformer_hmi_svg(vals["charge"], vals["oil"], vals["ambient"], vals["humidity"], vals["voltage"], label, color), height=380, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_diag:
        st.plotly_chart(plotly_gauge(score, color), use_container_width=True, config={"displayModeBar": False})
        if fault_label:
            fd = FAULT_DIRECTIVES[fault_label]
            st.markdown(f"""
            <div class="fault-card" style="border-left:3px solid {fd['color']};">
                <span class="fault-tag" style="background:{fd['color']}22;color:{fd['color']};">⚠ {fault_label}</span>
                <div style="color:{fd['color']};font-size:0.84rem;font-weight:600;margin-top:6px;">{urgent_prognosis(fault_label, fault_sev)}</div>
                <div style="margin-top:8px;font-size:0.68rem;color:#8B97AC;text-transform:uppercase;">🔧 Corriger</div>
                <div style="font-size:0.86rem;color:#DCE2EC;margin:2px 0 6px;">{fd['correct']}</div>
                <div style="font-size:0.68rem;color:#8B97AC;text-transform:uppercase;">🔌 Actions réseau</div>
                <div style="font-size:0.86rem;color:#DCE2EC;margin-top:2px;">• {fd['network'][0]}<br>• {fd['network'][1]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="fault-card" style="border-left:3px solid #49B586;">
                <span class="fault-tag" style="background:#49B58622;color:#49B586;">✓ Fonctionnement normal</span>
            </div>
            """, unsafe_allow_html=True)

    # Courbe de l'évolution rejouée depuis le début du scénario
    ts = np.linspace(0, p, max(2, st.session_state.fault_t + 1))
    scores_ts = []
    for pi in ts:
        vi = scn["fn"](pi, amb_base, hum_base)
        si, _ = compute_risk(vi["charge"], vi["oil"], vi["ambient"], vi["humidity"], vi["voltage"])
        scores_ts.append(si)
    fig = go.Figure(go.Scatter(x=ts * DURATION, y=scores_ts, mode="lines", line=dict(color="#4FA3D1", width=2), fill="tozeroy", fillcolor="rgba(79,163,209,0.1)"))
    fig.add_hrect(y0=0, y1=30, fillcolor="#49B586", opacity=0.06, line_width=0)
    fig.add_hrect(y0=30, y1=65, fillcolor="#E8A23D", opacity=0.06, line_width=0)
    fig.add_hrect(y0=65, y1=100, fillcolor="#E0554F", opacity=0.06, line_width=0)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"title": "Étape de simulation", "color": "#8B97AC", "gridcolor": "#1B2436"},
        yaxis={"title": "Indice de risque", "color": "#8B97AC", "gridcolor": "#1B2436", "range": [0, 100]},
        font={"color": "#E9ECF2"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Cliquez sur ▶️ Lecture pour rejouer l'évolution automatiquement, ou faites défiler en pause via Réinitialiser + Lecture par étapes.")

# ============================================================================
# PAGE — COMPARAISON DES 3 TYPES D'INSTALLATION
# ============================================================================
elif page == "🏗️ Comparaison des types d'installation":
    st.title("Comparaison des types d'installation")
    ref("Section 3.5 — mêmes conditions, expositions différentes")
    st.caption("Trois transformateurs identiques, soumis aux mêmes conditions climatiques de base, réagissent différemment selon leur mode d'installation.")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        use_live = st.checkbox("Utiliser la météo en direct", value=True)
    with cc2:
        charge_cmp = st.slider("Charge commune (%)", 0, 150, 75)
    with cc3:
        season_cmp = st.radio("Saison", ["seche", "pluvieuse"], format_func=lambda s: "Sèche" if s == "seche" else "Pluvieuse", horizontal=True)

    if use_live:
        w = fetch_live_weather(12.3714, -1.5197)
        amb0, hum0 = w["temperature"], w["humidity"]
        st.caption(f"Conditions de base utilisées : {amb0:.1f}°C / {hum0:.0f}% d'humidité ({'météo en direct' if w['source']=='live' else 'estimation hors-ligne'}).")
    else:
        cc4, cc5 = st.columns(2)
        with cc4: amb0 = st.slider("Température ambiante de référence (°C)", 20, 45, 34)
        with cc5: hum0 = st.slider("Humidité de référence (%)", 10, 95, 30 if season_cmp == "seche" else 80)

    st.write("")
    cols = st.columns(3)
    rows_compare = []
    for i, (ttype, meta) in enumerate(TRANSFO_TYPES.items()):
        ambient, humidity, oil = apply_type(amb0, hum0, charge_cmp, ttype)
        score, contribs = compute_risk(charge_cmp, oil, ambient, humidity, 2, season_cmp)
        label, color = classify(score)
        rows_compare.append(dict(type=ttype, ambient=ambient, humidity=humidity, oil=oil, score=score, label=label))
        with cols[i]:
            st.markdown(f"""
            <div class="hmi-panel" style="border-color:{color}55;">
                <div style="font-size:1.7rem;text-align:center;">{meta['icon']}</div>
                <div style="text-align:center;font-weight:700;color:#E9ECF2;margin:4px 0;">{ttype}</div>
                <div style="text-align:center;font-family:monospace;font-weight:700;font-size:1.5rem;color:{color};">{score:.0f}</div>
                <div style="text-align:center;font-size:0.72rem;color:{color};margin-bottom:8px;">{label}</div>
                <table style="width:100%;font-size:0.78rem;color:#DCE2EC;">
                    <tr><td style="color:#8B97AC;">T° ambiante</td><td style="text-align:right;">{ambient:.1f} °C</td></tr>
                    <tr><td style="color:#8B97AC;">T° huile</td><td style="text-align:right;">{oil:.1f} °C</td></tr>
                    <tr><td style="color:#8B97AC;">Humidité</td><td style="text-align:right;">{humidity:.0f} %</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            st.caption(meta["note"])

    st.write("")
    st.markdown("##### Pourquoi ces écarts ?")
    df_cmp = pd.DataFrame(rows_compare)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_cmp["type"], y=df_cmp["oil"], name="T° huile (°C)", marker_color="#E8A23D"))
    fig.add_trace(go.Bar(x=df_cmp["type"], y=df_cmp["ambient"], name="T° ambiante ressentie (°C)", marker_color="#4FA3D1"))
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280,
        margin=dict(l=10, r=10, t=20, b=10), font={"color": "#E9ECF2"},
        xaxis={"color": "#8B97AC"}, yaxis={"color": "#8B97AC", "gridcolor": "#1B2436"},
        legend={"orientation": "h", "y": 1.15},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("""
    <div class="section-note">
    À charge et climat de base identiques, la <b>cabine maçonnée</b> accumule le plus de chaleur
    interne faute de ventilation naturelle ; le <b>préfabriqué</b> sert de comportement de
    référence ; le <b>haut de poteau</b> subit les écarts climatiques les plus marqués
    (soleil, pluie directe) mais bénéficie d'un meilleur refroidissement par circulation d'air.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE — CARTE, HISTORIQUE & PRÉVISION DU PARC
# ============================================================================
elif page == "🗺️ Carte, historique & prévision du parc":
    st.title("Supervision du parc de transformateurs")
    ref("Section 3.9")
    st.caption("Vue d'ensemble numérotée et localisée, avec horloge de simulation, historique complet et prévision par tendance — dans l'esprit d'une intégration SCADA.")

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])
    with ctrl1:
        label_btn = "⏸️ Pause" if st.session_state.running else "▶️ Lecture"
        if st.button(label_btn, use_container_width=True):
            st.session_state.running = not st.session_state.running
            st.session_state.last_tick = time.time()
    with ctrl2:
        if st.button("⏭️ +1 heure", use_container_width=True):
            advance_time(1)
    with ctrl3:
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.session_state.sim_hour = 0
            st.session_state.historique = []
            st.session_state.state_history = []
            st.session_state.running = False
    with ctrl4:
        refresh_interval = st.slider("Vitesse (secondes réelles ≈ 1 heure simulée)", 1, 10, 3)

    if st.session_state.running:
        time.sleep(refresh_interval)
        advance_time(1)
        _rerun()

    jours = st.session_state.sim_hour // 24
    heure_j = st.session_state.sim_hour % 24
    etat_txt = "▶️ en lecture" if st.session_state.running else "⏸️ en pause"
    st.markdown(f"""
    <div class="clock-banner">
        <div class="clock-time">🕐 Jour {jours + 1} — {heure_j:02d}h00</div>
        <div class="clock-sub">Heure simulée n°{st.session_state.sim_hour} · {etat_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    df_map = fleet_state_at(st.session_state.sim_hour)
    df_map["couleur_hex"] = df_map["risque"].apply(
        lambda s: "#E0554F" if s >= 65 else "#E8A23D" if s >= 30 else "#49B586"
    )
    df_map["icon_data"] = df_map["couleur_hex"].apply(
        lambda c: {"url": _ICON_CACHE[c], "width": 64, "height": 64, "anchorY": 58}
    )
    df_map["tooltip_text"] = df_map.apply(
        lambda r: f"{r['id']} — {r['type']}\n{r['lieu']}\nRisque : {r['risque']} ({r['statut']})", axis=1
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Normal", str((df_map["statut"] == "Normal").sum()), "🟢 transformateurs")
    with k2: kpi("Surveillance renforcée", str((df_map["statut"] == "Surveillance renforcée").sum()), "🟠 transformateurs")
    with k3: kpi("Critique", str((df_map["statut"] == "Critique").sum()), "🔴 transformateurs")
    with k4: kpi("Pannes enregistrées", str(len(st.session_state.historique)), "depuis le début de la simulation")

    df_obstacles = pd.DataFrame([
        dict(id=f["id"], lat=f["obs_lat"], lon=f["obs_lon"], icon=f["obstacle_icon"],
             label=f["obstacle_label"], detail=f["obstacle_detail"])
        for f in FLEET_INFO
    ])
    _obst_icon_cache = {icon: emoji_icon_data_uri(icon) for icon in df_obstacles["icon"].unique()}
    df_obstacles["icon_data"] = df_obstacles["icon"].apply(
        lambda c: {"url": _obst_icon_cache[c], "width": 48, "height": 48, "anchorY": 44}
    )
    df_obstacles["tooltip_text"] = df_obstacles.apply(
        lambda r: f"{r['icon']} {r['label']} (près de {r['id']})\n{r['detail']}", axis=1
    )

    layer = pdk.Layer(
        "IconLayer", data=df_map, get_icon="icon_data",
        get_position=["lon", "lat"], get_size=4, size_scale=16,
        pickable=True,
    )
    obstacle_layer = pdk.Layer(
        "IconLayer", data=df_obstacles, get_icon="icon_data",
        get_position=["lon", "lat"], get_size=3, size_scale=14,
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=float(df_map["lat"].mean()), longitude=float(df_map["lon"].mean()), zoom=11.2, pitch=0,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[obstacle_layer, layer], initial_view_state=view_state,
            map_provider="carto", map_style="dark_matter",
            tooltip={"text": "{tooltip_text}"},
        ),
        use_container_width=True, height=520,
    )
    st.caption("⚡ Icônes colorées = transformateurs (couleur = statut de risque) · icônes rondes = obstacles à proximité (🏢 bâtiments, 🌳 arbre, 🌊 barrage) utiles pour préparer une intervention.")

    hist_df = pd.DataFrame(st.session_state.state_history) if st.session_state.state_history else pd.DataFrame(columns=["heure_sim", "id", "risque"])
    df_map["delai_estime"] = df_map.apply(
        lambda r: (trend_forecast(hist_df, r["id"], r["risque"]) or (estimate_delay(r["risque"])[0] or "—")), axis=1
    )
    df_map["numero"] = df_map["id"]
    obst_by_id = {f["id"]: f"{f['obstacle_icon']} {f['obstacle_label']} — {f['obstacle_detail']}" for f in FLEET_INFO}
    df_map["obstacles"] = df_map["id"].map(obst_by_id)

    st.markdown("##### Fiche par transformateur — numéro, type, localisation, risque, obstacles")
    st.dataframe(
        df_map[["numero", "type", "lieu", "charge", "oil", "ambient", "humidity", "risque", "statut", "delai_estime", "obstacles"]]
        .sort_values("risque", ascending=False)
        .rename(columns={
            "numero": "N°", "type": "Type d'installation", "lieu": "Localisation",
            "charge": "Charge (%)", "oil": "T° huile (°C)", "ambient": "T° ambiante (°C)",
            "humidity": "Humidité (%)", "risque": "Indice de risque", "statut": "Statut",
            "delai_estime": "Délai / tendance avant défaillance", "obstacles": "Obstacle à proximité",
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Historique des pannes critiques")
    ref("Journal cumulé au fil de la simulation")
    if st.session_state.historique:
        df_hist = pd.DataFrame(st.session_state.historique)
        df_hist_cum = (
            df_hist.groupby("heure_sim").size().reindex(
                range(0, st.session_state.sim_hour + 1), fill_value=0
            ).cumsum().reset_index()
        )
        df_hist_cum.columns = ["heure_sim", "pannes_cumulees"]
        fig_h = go.Figure(go.Scatter(
            x=df_hist_cum["heure_sim"], y=df_hist_cum["pannes_cumulees"],
            mode="lines", line=dict(color="#E0554F", width=2), fill="tozeroy",
            fillcolor="rgba(224,85,79,0.12)",
        ))
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            xaxis={"title": "Heure simulée", "color": "#8B97AC", "gridcolor": "#1B2436"},
            yaxis={"title": "Pannes cumulées", "color": "#8B97AC", "gridcolor": "#1B2436"},
            font={"color": "#E9ECF2"},
        )
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(
            df_hist.sort_values("heure_sim", ascending=False).rename(columns={
                "heure_sim": "Heure simulée", "id": "Transformateur",
                "risque": "Indice de risque", "charge": "Charge (%)", "oil": "T° huile (°C)",
            }),
            use_container_width=True, hide_index=True, height=260,
        )
        st.caption("La prévision affichée dans la fiche ci-dessus utilise cet historique complet (pas seulement les pannes) : une régression sur les derniers points de chaque transformateur estime le nombre d'heures avant d'atteindre le seuil critique si la tendance actuelle se maintient.")
    else:
        st.info("Aucune panne enregistrée pour l'instant. Cliquez sur ▶️ Lecture ou ⏭️ +1 heure pour avancer la simulation.")

# ============================================================================
# PAGE — DONNÉES RÉELLES (CSV)
# ============================================================================
elif page == "📄 Données réelles (CSV)":
    st.title("Appliquer le modèle à un fichier de mesures")
    ref("Chapitre 2 — données réelles")
    st.caption("Colonnes attendues : charge, oil, ambient, humidity, voltage, season (seche/pluvieuse)")

    file = st.file_uploader("Fichier CSV", type=["csv"])
    if file is not None:
        df_up = pd.read_csv(file)
        required = {"charge", "oil", "ambient", "humidity", "voltage"}
        if not required.issubset(df_up.columns):
            st.error(f"Colonnes manquantes. Attendu au minimum : {sorted(required)}")
        else:
            if "season" not in df_up.columns:
                df_up["season"] = "seche"
            df_up["indice_risque"] = df_up.apply(
                lambda r: compute_risk(r.charge, r.oil, r.ambient, r.humidity, r.voltage, r.season)[0], axis=1
            ).round(1)
            df_up["classification"] = df_up["indice_risque"].apply(lambda s: classify(s)[0])
            df_up["type_panne_probable"] = df_up.apply(
                lambda r: diagnose_fault(r.charge, r.oil, r.ambient, r.humidity, r.voltage)[0] or "Aucune", axis=1
            )

            k1, k2, k3 = st.columns(3)
            with k1: kpi("Enregistrements traités", str(len(df_up)))
            with k2: kpi("Risque moyen", f"{df_up['indice_risque'].mean():.1f} / 100")
            with k3: kpi("Cas critiques", str((df_up["classification"] == "Critique").sum()))

            st.dataframe(df_up, use_container_width=True)
            fig = go.Figure(go.Histogram(x=df_up["indice_risque"], marker_color="#4FA3D1", nbinsx=20))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis={"title": "Indice de risque", "color": "#8B97AC", "gridcolor": "#1B2436"},
                yaxis={"title": "Nombre d'enregistrements", "color": "#8B97AC", "gridcolor": "#1B2436"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.download_button(
                "Télécharger les résultats (CSV)",
                df_up.to_csv(index=False).encode("utf-8"),
                "resultats_prediction.csv", "text/csv",
            )
    else:
        st.write("Importez un fichier pour appliquer le modèle à vos propres données.")
