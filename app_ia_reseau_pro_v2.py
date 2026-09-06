"""
Système intelligent de prédiction de défaillance des transformateurs de distribution
--------------------------------------------------------------------------------------
MAQUETTE STREAMLIT — à connecter à votre vrai modèle entraîné (chapitre 2).

La fonction `compute_risk()` ci-dessous utilise une formule pondérée simplifiée,
construite uniquement pour permettre de tester et de présenter le pipeline
(capteurs -> prétraitement -> modèle -> interprétation) avant que le modèle réel
ne soit disponible. Pour brancher votre vrai modèle :

    import joblib
    model = joblib.load("mon_modele.pkl")
    proba = model.predict_proba(X)[:, 1] * 100   # remplace compute_risk()

Lancer localement :
    pip install streamlit pandas numpy pydeck
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(
    page_title="Prédiction de défaillance des transformateurs — Burkina Faso",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Moteur de risque (à remplacer par le modèle entraîné du chapitre 2)
# ----------------------------------------------------------------------------

def _clamp01(x):
    return max(0.0, min(1.0, x))


def compute_risk(charge, oil, ambient, humidity, voltage, season="seche"):
    """Retourne (score_total, contributions_par_variable)."""
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
        return "Normal", "🟢", "#49B586"
    elif score < 65:
        return "Surveillance renforcée", "🟠", "#E8A23D"
    else:
        return "Critique", "🔴", "#E0554F"


def explain(v, contribs, season):
    sorted_c = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
    total = min(100.0, sum(contribs.values()))
    if total < 30:
        return "Les valeurs transmises restent dans les plages de fonctionnement normal ; aucun facteur ne présente de contribution significative au risque."
    top, second = sorted_c[0], sorted_c[1]
    text = f"Le risque provient principalement de **{top[0].lower()}** ({v[top[0]]})"
    if second[1] > 3:
        text += f", combinée à **{second[0].lower()}** ({v[second[0]]})"
    if season == "pluvieuse":
        text += ", dans un contexte de saison pluvieuse qui accentue l'effet de l'humidité sur l'isolation"
    return text + "."


PRESETS = {
    "Fonctionnement normal — saison sèche": dict(charge=55, oil=58, ambient=33, humidity=22, voltage=2, season="seche"),
    "Pic de chaleur": dict(charge=85, oil=88, ambient=43, humidity=18, voltage=3, season="seche"),
    "Saison pluvieuse": dict(charge=65, oil=62, ambient=27, humidity=87, voltage=-2, season="pluvieuse"),
}

st.title("Système intelligent de prédiction — transformateurs de distribution")
st.caption(
    "Maquette de démonstration du pipeline capteurs → prétraitement → modèle IA → interprétation. "
    "Le moteur de risque utilisé ici est une formule pondérée simplifiée ; remplacez `compute_risk()` "
    "par votre modèle réellement entraîné (chapitre 2) avant tout usage en soutenance comme résultat final."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Simulation manuelle",
    "Série temporelle",
    "Comparaison des modèles",
    "Carte du parc",
    "Importer des données",
])

# ----------------------------------------------------------------------------
# Onglet 1 — Simulation manuelle
# ----------------------------------------------------------------------------
with tab1:
    col_ctrl, col_res = st.columns([1, 1.2])

    with col_ctrl:
        st.subheader("Capteurs du transformateur")

        preset_name = st.selectbox("Scénario préconstruit", ["— Réglage manuel —"] + list(PRESETS.keys()))
        defaults = PRESETS.get(preset_name, dict(charge=60, oil=58, ambient=33, humidity=22, voltage=2, season="seche"))

        charge = st.slider("Charge (% de la puissance nominale)", 0, 150, defaults["charge"])
        oil = st.slider("Température de l'huile (°C)", 30, 120, defaults["oil"])
        ambient = st.slider("Température ambiante (°C)", 15, 45, defaults["ambient"])
        humidity = st.slider("Humidité relative (%)", 10, 95, defaults["humidity"])
        voltage = st.slider("Écart de tension par rapport au nominal (%)", -15, 15, defaults["voltage"])
        season = st.radio("Saison", ["seche", "pluvieuse"], format_func=lambda s: "Sèche" if s == "seche" else "Pluvieuse",
                           index=0 if defaults["season"] == "seche" else 1, horizontal=True)

        run = st.button("Transmettre au modèle IA", type="primary", use_container_width=True)

    with col_res:
        st.subheader("Interprétation du modèle")
        if run or preset_name != "— Réglage manuel —":
            with st.spinner("Prétraitement puis calcul de l'indice de risque..."):
                time.sleep(0.4)
            score, contribs = compute_risk(charge, oil, ambient, humidity, voltage, season)
            label, emoji, color = classify(score)

            m1, m2 = st.columns(2)
            m1.metric("Indice de risque", f"{score:.0f} / 100")
            m2.markdown(
                f"<div style='padding:10px 14px;border-radius:8px;background:{color}22;"
                f"color:{color};font-weight:600;text-align:center;margin-top:6px;'>{emoji} {label}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**Facteurs contribuant au risque**")
            v_display = {
                "Charge": f"{charge} %", "Température huile": f"{oil} °C",
                "Température ambiante": f"{ambient} °C", "Humidité": f"{humidity} %",
                "Écart de tension": f"{voltage:+d} %",
            }
            df_c = pd.DataFrame({"Facteur": list(contribs.keys()), "Contribution": list(contribs.values())})
            df_c = df_c.sort_values("Contribution", ascending=True)
            st.bar_chart(df_c.set_index("Facteur"))

            st.info(explain(v_display, contribs, season))
        else:
            st.write("Ajustez les capteurs ou choisissez un scénario, puis transmettez au modèle.")

# ----------------------------------------------------------------------------
# Onglet 2 — Série temporelle (simulation dynamique)
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Simulation d'un cycle de fonctionnement (48 heures)")
    st.caption("Cycle jour/nuit sur la température et l'humidité, avec option de surcharge progressive — illustre la détection de dérive (section 3.6).")

    c1, c2 = st.columns(2)
    overload = c1.checkbox("Simuler une surcharge progressive à partir de h=24", value=True)
    season_ts = c2.radio("Saison", ["seche", "pluvieuse"], format_func=lambda s: "Sèche" if s == "seche" else "Pluvieuse", horizontal=True, key="season_ts")

    hours = np.arange(0, 48)
    ambient_ts = 30 + 10 * np.sin((hours - 9) / 24 * 2 * np.pi) + (5 if season_ts == "seche" else -3)
    humidity_ts = 25 - 8 * np.sin((hours - 9) / 24 * 2 * np.pi) if season_ts == "seche" else 75 + 10 * np.sin((hours - 6) / 24 * 2 * np.pi)
    base_charge = 50 + 25 * np.clip(np.sin((hours - 7) / 24 * 2 * np.pi), 0, None)
    charge_ts = base_charge.copy()
    if overload:
        ramp = np.clip((hours - 24) / 24, 0, 1) * 70
        charge_ts = charge_ts + ramp
    oil_ts = 45 + 0.5 * charge_ts + 0.3 * (ambient_ts - 30)
    voltage_ts = np.random.default_rng(0).normal(0, 3, size=len(hours))

    scores = []
    for i in range(len(hours)):
        s, _ = compute_risk(charge_ts[i], oil_ts[i], ambient_ts[i], humidity_ts[i], voltage_ts[i], season_ts)
        scores.append(s)

    df_ts = pd.DataFrame({
        "Heure": hours, "Charge (%)": charge_ts, "T° huile (°C)": oil_ts,
        "T° ambiante (°C)": ambient_ts, "Humidité (%)": humidity_ts, "Indice de risque": scores,
    }).set_index("Heure")

    play = st.button("Lancer l'animation")
    chart_ph = st.empty()
    metric_ph = st.empty()

    if play:
        for i in range(2, len(hours) + 1):
            chart_ph.line_chart(df_ts[["Indice de risque"]].iloc[:i])
            last_score = df_ts["Indice de risque"].iloc[i - 1]
            label, emoji, color = classify(last_score)
            metric_ph.markdown(f"**Heure {hours[i-1]}** — indice {last_score:.0f}/100 — {emoji} {label}")
            time.sleep(0.05)
    else:
        chart_ph.line_chart(df_ts[["Indice de risque"]])

    with st.expander("Voir les variables brutes simulées"):
        st.line_chart(df_ts[["Charge (%)", "T° huile (°C)", "T° ambiante (°C)"]])
        st.dataframe(df_ts, use_container_width=True)

# ----------------------------------------------------------------------------
# Onglet 3 — Comparaison des modèles (chapitre 2)
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Comparaison des modèles entraînés")
    st.warning(
        "Valeurs d'exemple à remplacer par vos résultats réels une fois les modèles entraînés (section 2.9)."
    )
    df_models = pd.DataFrame({
        "Modèle": ["Régression logistique", "Random Forest", "XGBoost", "SVM", "Réseau de neurones"],
        "Précision": [0.71, 0.86, 0.88, 0.79, 0.84],
        "Rappel": [0.65, 0.83, 0.85, 0.74, 0.80],
        "Score F1": [0.68, 0.84, 0.86, 0.76, 0.82],
        "AUC-ROC": [0.74, 0.90, 0.92, 0.83, 0.88],
    })
    st.dataframe(df_models, use_container_width=True, hide_index=True)
    metric_choice = st.selectbox("Comparer selon", ["Score F1", "AUC-ROC", "Précision", "Rappel"])
    st.bar_chart(df_models.set_index("Modèle")[[metric_choice]])
    best = df_models.loc[df_models[metric_choice].idxmax(), "Modèle"]
    st.success(f"Modèle le plus performant selon **{metric_choice}** : **{best}**")

# ----------------------------------------------------------------------------
# Onglet 4 — Carte du parc de transformateurs
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Supervision du parc de transformateurs")
    st.caption("Exemple de vue d'ensemble pour plusieurs postes — illustre l'intégration envisagée au SCADA (section 3.9).")

    rng = np.random.default_rng(42)
    n = 14
    base_lat, base_lon = 12.3714, -1.5197  # Ouagadougou
    df_map = pd.DataFrame({
        "id": [f"TR-{i+1:03d}" for i in range(n)],
        "lat": base_lat + rng.normal(0, 0.05, n),
        "lon": base_lon + rng.normal(0, 0.05, n),
        "charge": rng.integers(30, 140, n),
        "oil": rng.integers(45, 105, n),
        "ambient": rng.integers(25, 42, n),
        "humidity": rng.integers(15, 85, n),
        "voltage": rng.integers(-10, 10, n),
    })
    risks = df_map.apply(lambda r: compute_risk(r.charge, r.oil, r.ambient, r.humidity, r.voltage)[0], axis=1)
    df_map["risque"] = risks
    df_map["couleur"] = df_map["risque"].apply(lambda s: [224, 85, 79] if s >= 65 else [232, 162, 61] if s >= 30 else [73, 181, 134])

    layer = pdk.Layer(
        "ScatterplotLayer", data=df_map, get_position=["lon", "lat"],
        get_fill_color="couleur", get_radius=350, pickable=True,
    )
    view_state = pdk.ViewState(latitude=base_lat, longitude=base_lon, zoom=10)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                              tooltip={"text": "{id}\nRisque : {risque}"}))
    st.dataframe(df_map[["id", "charge", "oil", "ambient", "humidity", "risque"]].round(1), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Onglet 5 — Importer un CSV de mesures réelles
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Appliquer le modèle à un fichier de mesures")
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
            results = df_up.apply(lambda r: compute_risk(r.charge, r.oil, r.ambient, r.humidity, r.voltage, r.season)[0], axis=1)
            df_up["indice_risque"] = results.round(1)
            df_up["classification"] = df_up["indice_risque"].apply(lambda s: classify(s)[0])
            st.dataframe(df_up, use_container_width=True)
            st.bar_chart(df_up["classification"].value_counts())
            st.download_button(
                "Télécharger les résultats (CSV)",
                df_up.to_csv(index=False).encode("utf-8"),
                "resultats_prediction.csv",
                "text/csv",
            )
    else:
        st.write("Importez un fichier pour appliquer le modèle à vos propres données.")
