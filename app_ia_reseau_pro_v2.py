"""
Système intelligent de prédiction de défaillance des transformateurs de distribution
--------------------------------------------------------------------------------------
Tableau de bord de démonstration reliant explicitement les éléments du mémoire :
- Architecture en couches (section 3.2)
- Pipeline capteurs -> prétraitement -> modèle -> interprétation (section 3.4)
- Simulation des conditions de fonctionnement (section 3.5)
- Détection de dérive / risque progressif (section 3.6)
- Console opérateur : schéma du transformateur, capteurs et interprétation en direct (section 3.4)
- Intégration SCADA / supervision du parc avec horloge de simulation et historique
  des pannes (section 3.9)

La fonction `compute_risk()` utilise une formule pondérée simplifiée, construite pour
tester et présenter le pipeline avant que le vrai modèle entraîné ne soit disponible.
Pour brancher le modèle réel :

    import joblib
    model = joblib.load("mon_modele.pkl")
    proba = model.predict_proba(X)[:, 1] * 100   # remplace compute_risk()

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
    .block-container {padding-top: 1.6rem;}
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
    .hmi-readout-label{
        font-family:'JetBrains Mono','IBM Plex Mono',monospace;font-size:0.68rem;
        color:#8B97AC;letter-spacing:0.05em;text-transform:uppercase;
    }
    .hmi-readout-value{
        font-family:'JetBrains Mono','IBM Plex Mono',monospace;font-size:1.3rem;
        font-weight:700;
    }
    .hmi-interpret{
        background:#141C2A;border:1px solid #26314A;border-left:3px solid #4FA3D1;
        border-radius:6px;padding:12px 16px;font-size:0.92rem;color:#DCE2EC;line-height:1.55;
    }
</style>
""", unsafe_allow_html=True)


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

    w = {"charge": 35, "oil": 30, "ambient": 15, "humidity": 15, "voltage": 5}
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


# ----------------------------------------------------------------------------
# PRONOSTIC : délai avant panne estimé et démarche corrective
# ----------------------------------------------------------------------------
# Estimation illustrative fondée sur le niveau de l'indice de risque — à
# remplacer, une fois le modèle entraîné disponible, par une estimation issue
# d'une analyse de survie / durée de vie résiduelle (RUL) sur données réelles.

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


CORRECTIVE_ACTIONS = {
    "Charge": "Délester une partie de la charge vers un poste voisin ou lisser les pointes de consommation ; "
              "si la surcharge est récurrente, étudier un renforcement de puissance.",
    "Température huile": "Contrôler le système de refroidissement (radiateurs, ventilateurs), vérifier le niveau "
                          "et la qualité de l'huile, et s'assurer qu'aucun encrassement ne bloque la dissipation thermique.",
    "Température ambiante": "Améliorer la ventilation du poste ou du local, protéger le transformateur de "
                             "l'ensoleillement direct, et limiter la charge aux heures les plus chaudes.",
    "Humidité": "Vérifier l'étanchéité de la cuve et des joints, contrôler ou remplacer le gel de silice du "
                "respirateur, et envisager un séchage de l'huile si l'humidité interne est confirmée.",
    "Écart de tension": "Vérifier le régulateur de tension et l'équilibrage des phases, contrôler les prises "
                         "du changeur de régulation, et signaler tout déséquilibre au poste source.",
}


def prognosis(score, contribs):
    """Retourne (délai estimé, urgence, facteur dominant, action corrective)
    ou None si le transformateur est en fonctionnement normal."""
    delay, urgency = estimate_delay(score)
    if delay is None:
        return None
    top_factor = max(contribs.items(), key=lambda kv: kv[1])[0]
    action = CORRECTIVE_ACTIONS.get(top_factor, "Effectuer une inspection technique du transformateur.")
    return delay, urgency, top_factor, action


PRESETS = {
    "Fonctionnement normal — saison sèche": dict(charge=55, oil=58, ambient=33, humidity=22, voltage=2, season="seche"),
    "Pic de chaleur": dict(charge=85, oil=88, ambient=43, humidity=18, voltage=3, season="seche"),
    "Saison pluvieuse": dict(charge=65, oil=62, ambient=27, humidity=87, voltage=-2, season="pluvieuse"),
    "Surcharge critique": dict(charge=138, oil=108, ambient=40, humidity=30, voltage=-9, season="seche"),
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
        ("Couche d'acquisition", "Capteurs, compteurs intelligents, API climatiques"),
        ("Couche de traitement des données", "Nettoyage, normalisation, mise en forme"),
        ("Couche d'intelligence", "Modèle entraîné — calcul de l'indice de risque"),
        ("Couche applicative", "Comparaison aux seuils, génération des alertes"),
        ("Couche de présentation", "Tableau de bord, notifications, supervision"),
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
    return f"""
    <div style="font-family:'IBM Plex Sans',sans-serif;max-width:520px;">{rows}</div>
    """


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
    """Code couleur individuel d'un capteur, indépendant de l'indice global,
    pour un affichage de type poste de supervision (chaque mesure a son propre
    seuil d'alerte)."""
    v = abs(value) if absolute else value
    if v >= crit:
        return "#E0554F"
    if v >= warn:
        return "#E8A23D"
    return "#49B586"


def transformer_hmi_svg(charge, oil, ambient, humidity, voltage, status_label, status_color):
    """Schéma de supervision (mimic diagram) du transformateur : chaque capteur
    est représenté à l'endroit physique où il est mesuré, avec sa valeur
    instantanée et son propre code couleur — dans l'esprit d'un écran opérateur
    SCADA, tout en restant lisible pour un public non spécialiste."""
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

      <!-- Traverse haute + isolateurs -->
      <line x1="270" y1="45" x2="410" y2="45" stroke="{c_charge}" stroke-width="3"/>
      <line x1="300" y1="45" x2="300" y2="80" stroke="{c_charge}" stroke-width="3"/>
      <line x1="380" y1="45" x2="380" y2="80" stroke="{c_volt}" stroke-width="3"/>
      <circle cx="300" cy="80" r="7" fill="none" stroke="{c_charge}" stroke-width="3"/>
      <circle cx="380" cy="80" r="7" fill="none" stroke="{c_volt}" stroke-width="3"/>

      <!-- Cuve principale -->
      <rect x="255" y="95" width="170" height="130" rx="12" fill="{status_color}" fill-opacity="0.14" stroke="{status_color}" stroke-width="3.5"/>
      <circle cx="305" cy="160" r="26" fill="none" stroke="{status_color}" stroke-width="2.6"/>
      <circle cx="375" cy="160" r="26" fill="none" stroke="{status_color}" stroke-width="2.6"/>

      <!-- Ailettes de refroidissement (côté T° huile) -->
      <line x1="425" y1="115" x2="445" y2="115" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="140" x2="445" y2="140" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="165" x2="445" y2="165" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="190" x2="445" y2="190" stroke="{c_oil}" stroke-width="4"/>
      <line x1="425" y1="210" x2="445" y2="210" stroke="{c_oil}" stroke-width="4"/>

      <!-- Mise à la terre -->
      <line x1="340" y1="225" x2="340" y2="255" stroke="#8B97AC" stroke-width="3"/>
      <line x1="322" y1="255" x2="358" y2="255" stroke="#8B97AC" stroke-width="3"/>
      <line x1="327" y1="261" x2="353" y2="261" stroke="#8B97AC" stroke-width="2.2"/>
      <line x1="332" y1="267" x2="348" y2="267" stroke="#8B97AC" stroke-width="1.6"/>

      <!-- Repère -->
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


# ============================================================================
# ICÔNE DE TRANSFORMATEUR (pictogramme réaliste, colorée selon le statut)
# ============================================================================

def transformer_icon_data_uri(color):
    """Génère un pictogramme de transformateur de distribution (cuve, isolateurs,
    mise à la terre) encodé en SVG data-URI, coloré selon le niveau de risque."""
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


_ICON_CACHE = {
    "#E0554F": transformer_icon_data_uri("#E0554F"),
    "#E8A23D": transformer_icon_data_uri("#E8A23D"),
    "#49B586": transformer_icon_data_uri("#49B586"),
}

# ============================================================================
# ÉTAT DE LA SIMULATION DU PARC (horloge + historique des pannes)
# ============================================================================

FLEET_SIZE = 5
FLEET_BASE_LAT, FLEET_BASE_LON = 12.3714, -1.5197  # Ouagadougou

if "sim_hour" not in st.session_state:
    st.session_state.sim_hour = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "historique" not in st.session_state:
    st.session_state.historique = []
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()
if "fleet_base" not in st.session_state:
    rng0 = np.random.default_rng(42)
    st.session_state.fleet_base = pd.DataFrame({
        "id": [f"TR-{i+1:03d}" for i in range(FLEET_SIZE)],
        "lat": FLEET_BASE_LAT + rng0.normal(0, 0.05, FLEET_SIZE),
        "lon": FLEET_BASE_LON + rng0.normal(0, 0.05, FLEET_SIZE),
        "base_charge": rng0.uniform(45, 75, FLEET_SIZE),
        "fault_prone": rng0.uniform(0, 1, FLEET_SIZE) < 0.3,
    })


def fleet_state_at(hour):
    """Calcule l'état (capteurs + risque) de chaque transformateur du parc à
    une heure simulée donnée. Déterministe par heure -> reproductible si on
    revient en arrière, tout en variant dans le temps (cycle jour/nuit,
    saison, événements de surcharge occasionnels sur les transfos fragiles)."""
    base = st.session_state.fleet_base
    n = len(base)
    day_frac = hour % 24
    season = "pluvieuse" if (hour // 24) % 5 == 4 else "seche"

    ambient = 30 + 10 * np.sin((day_frac - 9) / 24 * 2 * np.pi) + (5 if season == "seche" else -3)
    humidity_center = 25 if season == "seche" else 78

    rng_h = np.random.default_rng(hour)  # même heure -> même tirage
    charge = base["base_charge"].values + 25 * max(0.0, np.sin((day_frac - 7) / 24 * 2 * np.pi))
    overload = (rng_h.uniform(0, 1, n) < 0.04) & base["fault_prone"].values
    charge = charge + overload * rng_h.uniform(40, 85, n)

    oil = 45 + 0.5 * charge + 0.3 * (ambient - 30)
    humidity = humidity_center + rng_h.normal(0, 5, n)
    voltage = rng_h.normal(0, 4, n)

    df = base.copy()
    df["charge"] = np.round(charge, 1)
    df["oil"] = np.round(oil, 1)
    df["ambient"] = round(float(ambient), 1)
    df["humidity"] = np.round(np.clip(humidity, 5, 98), 1)
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
        for r in df[df["statut"] == "Critique"].itertuples():
            st.session_state.historique.append({
                "heure_sim": st.session_state.sim_hour,
                "id": r.id,
                "risque": r.risque,
                "charge": r.charge,
                "oil": r.oil,
            })
    if len(st.session_state.historique) > 800:
        st.session_state.historique = st.session_state.historique[-800:]


# ============================================================================
# NAVIGATION
# ============================================================================

st.sidebar.title("⚡ Système de prédiction")
st.sidebar.caption("Transformateurs de distribution — Burkina Faso")
page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Console opérateur",
    "Cycle de fonctionnement (48h)",
    "Carte du parc",
    "Données réelles (CSV)",
])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Le moteur de risque utilisé dans ce prototype est une formule pondérée illustrative. "
    "Remplacez `compute_risk()` par le modèle réellement entraîné (chapitre 2) avant toute "
    "présentation comme résultat final."
)

# ============================================================================
# PAGE — VUE D'ENSEMBLE
# ============================================================================
if page == "Vue d'ensemble":
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:2px;">
        <div style="font-size:2.1rem;">⚡</div>
        <div>
            <div style="font-size:1.7rem;font-weight:700;color:#E9ECF2;line-height:1.2;">
                Système intelligent de prédiction de défaillance
            </div>
            <div style="font-size:0.88rem;color:#8B97AC;margin-top:2px;">
                Supervision de 5 transformateurs de distribution — Ouagadougou
                <span class="ref-badge">Chapitre 3</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Transformateurs surveillés", str(FLEET_SIZE), "parc d'étude")
    with c2: kpi("Points de mesure", "5", "par transformateur, temps réel")
    with c3: kpi("Niveaux d'alerte", "3", "normal / surveillance / critique")
    with c4: kpi("Horloge de simulation", f"h+{st.session_state.sim_hour}", "voir Carte du parc")

    st.write("")
    st.markdown("##### État instantané du parc")
    ref("Section 3.9 — snapshot à l'heure simulée actuelle")
    df_snapshot = fleet_state_at(st.session_state.sim_hour)
    snap_cols = st.columns(FLEET_SIZE)
    for i, r in enumerate(df_snapshot.itertuples()):
        stlabel, stcolor = classify(r.risque)
        with snap_cols[i]:
            st.markdown(f"""
            <div class="hmi-panel" style="text-align:center;border-color:{stcolor}55;">
                <div style="font-family:monospace;font-size:0.78rem;color:#8B97AC;">{r.id}</div>
                <div style="font-size:1.5rem;margin:4px 0;">⚡</div>
                <div style="font-family:monospace;font-weight:700;font-size:1.1rem;color:{stcolor};">{r.risque:.0f}</div>
                <div style="font-size:0.68rem;color:{stcolor};">{stlabel}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("Valeurs recalculées automatiquement à partir de l'horloge de simulation (page « Carte du parc »).")

    st.write("")
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.markdown("##### Architecture du système")
        ref("Section 3.2")
        st.markdown('<div class="hmi-frame">', unsafe_allow_html=True)
        components.html(architecture_visual_html(), height=340, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("##### Pipeline de traitement")
        ref("Section 3.4")
        st.markdown('<div class="hmi-frame">', unsafe_allow_html=True)
        components.html(pipeline_visual_html(), height=210, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-note">
        Chaque mesure capteur traverse successivement le prétraitement (nettoyage,
        normalisation), le modèle d'intelligence artificielle entraîné, puis la couche
        applicative qui génère l'indice de risque, l'alerte et — au-delà d'un certain
        seuil — un délai estimé avant panne et une démarche corrective (voir « Console
        opérateur »).
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Comment naviguer ce tableau de bord")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**🖥️ Console opérateur**")
        st.caption("Réglez les capteurs d'un transformateur : schéma en direct, interprétation, délai estimé avant panne et démarche corrective.")
    with d2:
        st.markdown("**📈 Cycle 48h**")
        st.caption("Rejoue un cycle jour/nuit avec option de surcharge progressive — détection de dérive (3.6).")
    with d3:
        st.markdown("**🗺️ Carte du parc**")
        st.caption("Supervision des 5 transformateurs avec horloge de simulation et historique des pannes, dans l'esprit d'une intégration SCADA (3.9).")

# ============================================================================
# PAGE — SIMULATION EN DIRECT (réactive, sans bouton)
# ============================================================================
elif page == "Console opérateur":
    st.title("Console opérateur — Transformateur")
    ref("Sections 3.4 – 3.5")
    st.caption("Écran de supervision d'un transformateur : chaque capteur est affiché à l'endroit physique où il est mesuré, avec son propre code couleur, comme sur un poste de contrôle réel.")

    st.markdown("##### Réglage des capteurs")
    sc1, sc2 = st.columns([1.4, 1])
    with sc1:
        preset_name = st.selectbox("Scénario préconstruit", ["— Réglage manuel —"] + list(PRESETS.keys()))
    with sc2:
        defaults = PRESETS.get(preset_name, dict(charge=60, oil=58, ambient=33, humidity=22, voltage=2, season="seche"))
        season = st.radio("Saison", ["seche", "pluvieuse"],
                           format_func=lambda s: "Sèche" if s == "seche" else "Pluvieuse",
                           index=0 if defaults["season"] == "seche" else 1, horizontal=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1: charge = st.slider("Charge (%)", 0, 150, defaults["charge"])
    with s2: oil = st.slider("T° huile (°C)", 30, 120, defaults["oil"])
    with s3: ambient = st.slider("T° ambiante (°C)", 15, 45, defaults["ambient"])
    with s4: humidity = st.slider("Humidité (%)", 10, 95, defaults["humidity"])
    with s5: voltage = st.slider("Écart tension (%)", -15, 15, defaults["voltage"])

    score, contribs = compute_risk(charge, oil, ambient, humidity, voltage, season)
    label, color = classify(score)
    v_display = {
        "Charge": f"{charge} %", "Température huile": f"{oil} °C",
        "Température ambiante": f"{ambient} °C", "Humidité": f"{humidity} %",
        "Écart de tension": f"{voltage:+d} %",
    }

    st.write("")
    st.markdown('<div class="hmi-frame">', unsafe_allow_html=True)
    st.markdown('<div class="hmi-title">▣ Schéma de supervision — mesures en temps réel</div>', unsafe_allow_html=True)
    components.html(transformer_hmi_svg(charge, oil, ambient, humidity, voltage, label, color), height=380, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("🟢 Normal · 🟠 Surveillance renforcée · 🔴 Critique — chaque encadré est coloré selon le seuil propre à cette mesure ; la cuve et le statut global suivent l'indice de risque combiné (section 3.6).")

    st.write("")
    col_interp, col_detail = st.columns([1.1, 1])
    with col_interp:
        st.markdown("##### Interprétation en langage clair")
        st.markdown(f'<div class="hmi-interpret">{explain(v_display, contribs, season)}</div>', unsafe_allow_html=True)

        prog = prognosis(score, contribs)
        if prog:
            delay, urgency, top_factor, action = prog
            st.write("")
            st.markdown(f"""
            <div style="background:#141C2A;border:1px solid #26314A;border-left:3px solid {color};
                        border-radius:6px;padding:12px 16px;">
                <div style="font-family:monospace;font-size:0.7rem;color:{color};text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:6px;">⚠ Pronostic — {urgency}</div>
                <div style="font-size:0.9rem;color:#DCE2EC;margin-bottom:10px;">
                    Si les conditions actuelles persistent, une panne liée à
                    <b>{top_factor.lower()}</b> est possible d'ici <b>{delay}</b>.
                </div>
                <div style="font-family:monospace;font-size:0.7rem;color:#8B97AC;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:6px;">🔧 Démarche corrective recommandée</div>
                <div style="font-size:0.9rem;color:#DCE2EC;">{action}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Estimation illustrative basée sur l'indice de risque actuel — à affiner avec une analyse de durée de vie résiduelle une fois le modèle entraîné sur données réelles (chapitre 2).")
        st.write("")
        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(plotly_gauge(score, color), use_container_width=True, config={"displayModeBar": False})
        with g2:
            st.markdown(f"""
            <div style="padding:12px 16px;border-radius:8px;background:{color}22;color:{color};
                        font-weight:700;text-align:center;margin-top:36px;">{label}</div>
            """, unsafe_allow_html=True)
            st.caption("Classification déduite des seuils définis en section 3.6.")
    with col_detail:
        st.markdown("##### Poids de chaque facteur dans l'indice")
        st.plotly_chart(plotly_factors(contribs, v_display), use_container_width=True, config={"displayModeBar": False})

# ============================================================================
# PAGE — CYCLE DE FONCTIONNEMENT 48H
# ============================================================================
elif page == "Cycle de fonctionnement (48h)":
    st.title("Simulation d'un cycle de fonctionnement")
    ref("Section 3.5 – 3.6")
    st.caption("Cycle jour/nuit sur température et humidité, avec option de surcharge progressive — illustre la détection de dérive.")

    c1, c2 = st.columns(2)
    overload = c1.checkbox("Simuler une surcharge progressive à partir de h=24", value=True)
    season_ts = c2.radio("Saison", ["seche", "pluvieuse"], format_func=lambda s: "Sèche" if s == "seche" else "Pluvieuse", horizontal=True)

    hours = np.arange(0, 48)
    ambient_ts = 30 + 10 * np.sin((hours - 9) / 24 * 2 * np.pi) + (5 if season_ts == "seche" else -3)
    humidity_ts = (25 - 8 * np.sin((hours - 9) / 24 * 2 * np.pi)) if season_ts == "seche" else (75 + 10 * np.sin((hours - 6) / 24 * 2 * np.pi))
    base_charge = 50 + 25 * np.clip(np.sin((hours - 7) / 24 * 2 * np.pi), 0, None)
    charge_ts = base_charge.copy()
    if overload:
        charge_ts = charge_ts + np.clip((hours - 24) / 24, 0, 1) * 70
    oil_ts = 45 + 0.5 * charge_ts + 0.3 * (ambient_ts - 30)
    voltage_ts = np.random.default_rng(0).normal(0, 3, size=len(hours))

    scores, colors = [], []
    for i in range(len(hours)):
        s, _ = compute_risk(charge_ts[i], oil_ts[i], ambient_ts[i], humidity_ts[i], voltage_ts[i], season_ts)
        scores.append(s)
        colors.append(classify(s)[1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=scores, mode="lines", line=dict(color="#4FA3D1", width=2), name="Indice de risque"))
    fig.add_trace(go.Scatter(x=hours, y=scores, mode="markers",
                              marker=dict(color=colors, size=6), showlegend=False))
    fig.add_hrect(y0=0, y1=30, fillcolor="#49B586", opacity=0.06, line_width=0)
    fig.add_hrect(y0=30, y1=65, fillcolor="#E8A23D", opacity=0.06, line_width=0)
    fig.add_hrect(y0=65, y1=100, fillcolor="#E0554F", opacity=0.06, line_width=0)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"title": "Heure", "color": "#8B97AC", "gridcolor": "#1B2436"},
        yaxis={"title": "Indice de risque", "color": "#8B97AC", "gridcolor": "#1B2436", "range": [0, 100]},
        font={"color": "#E9ECF2"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    peak_idx = int(np.argmax(scores))
    kc1, kc2, kc3 = st.columns(3)
    with kc1: kpi("Risque maximal atteint", f"{scores[peak_idx]:.0f} / 100", f"à l'heure {hours[peak_idx]}")
    with kc2: kpi("Heures en zone critique", f"{sum(1 for s in scores if s >= 65)} h", "sur 48 h simulées")
    with kc3: kpi("Charge maximale simulée", f"{charge_ts.max():.0f} %", "de la puissance nominale")

    with st.expander("Voir les variables brutes simulées"):
        df_ts = pd.DataFrame({
            "Heure": hours, "Charge (%)": charge_ts.round(1), "T° huile (°C)": oil_ts.round(1),
            "T° ambiante (°C)": ambient_ts.round(1), "Humidité (%)": humidity_ts.round(1),
            "Indice de risque": np.round(scores, 1),
        }).set_index("Heure")
        st.line_chart(df_ts[["Charge (%)", "T° huile (°C)", "T° ambiante (°C)"]])
        st.dataframe(df_ts, use_container_width=True)

# ============================================================================
# PAGE — CARTE DU PARC (avec horloge de simulation + historique des pannes)
# ============================================================================
elif page == "Carte du parc":
    st.title("Supervision du parc de transformateurs")
    ref("Section 3.9")
    st.caption("Vue d'ensemble multi-postes avec horloge de simulation — illustre l'intégration envisagée au système SCADA.")

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
            st.session_state.running = False
    with ctrl4:
        refresh_interval = st.slider("Vitesse (secondes réelles ≈ 1 heure simulée)", 1, 10, 3)

    # avance automatique si en lecture : on ne fait avancer le temps que si
    # l'intervalle réel choisi s'est bien écoulé, puis on programme un
    # rechargement de page pour la prochaine étape.
    if st.session_state.running:
        now = time.time()
        if now - st.session_state.last_tick >= refresh_interval:
            advance_time(1)
            st.session_state.last_tick = now
        components.html(f"""
        <script>
        setTimeout(function() {{ window.parent.location.reload(); }}, {int(refresh_interval * 1000)});
        </script>
        """, height=0)

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

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Normal", str((df_map["statut"] == "Normal").sum()), "🟢 transformateurs")
    with k2: kpi("Surveillance renforcée", str((df_map["statut"] == "Surveillance renforcée").sum()), "🟠 transformateurs")
    with k3: kpi("Critique", str((df_map["statut"] == "Critique").sum()), "🔴 transformateurs")
    with k4: kpi("Pannes enregistrées", str(len(st.session_state.historique)), "depuis le début de la simulation")

    layer = pdk.Layer(
        "IconLayer", data=df_map, get_icon="icon_data",
        get_position=["lon", "lat"], get_size=4, size_scale=16,
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=float(df_map["lat"].mean()), longitude=float(df_map["lon"].mean()), zoom=11.3, pitch=0,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer], initial_view_state=view_state,
            map_provider="carto", map_style="dark_matter",
            tooltip={"text": "{id}\nRisque : {risque}\nStatut : {statut}"},
        ),
        use_container_width=True, height=560,
    )

    df_map["delai_estime"] = df_map.apply(
        lambda r: (estimate_delay(r["risque"])[0] or "—"), axis=1
    )

    st.dataframe(
        df_map[["id", "charge", "oil", "ambient", "humidity", "risque", "statut", "delai_estime"]]
        .sort_values("risque", ascending=False)
        .rename(columns={"delai_estime": "Délai estimé avant panne"}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Historique des pannes / alertes critiques")
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
    else:
        st.info("Aucune panne enregistrée pour l'instant. Cliquez sur ▶️ Lecture ou ⏭️ +1 heure pour avancer la simulation.")

# ============================================================================
# PAGE — DONNÉES RÉELLES (CSV)
# ============================================================================
elif page == "Données réelles (CSV)":
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
