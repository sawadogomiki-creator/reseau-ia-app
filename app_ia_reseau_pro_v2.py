"""
Système intelligent de supervision et détection de défauts réseau.

Prototype de recherche combinant :
  - un score de risque continu (règles physiques, indépendant du ML)
  - un classifieur XGBoost multi-classes entraîné sur données synthétiques
  - un historique réel de mesures accumulé en session
  - un tableau de bord Streamlit pour l'exploration interactive

⚠️ Données synthétiques — à valider sur données réelles avant tout usage
industriel. Ne pas utiliser comme système de protection ou de commande.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ============================================================
# CONFIGURATION GÉNÉRALE
# ============================================================
st.set_page_config(
    page_title="IA - Supervision intelligente du réseau",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "Voltage (V)",
    "Current (A)",
    "Temperature (°C)",
    "Wind Speed (km/h)",
]

CLASS_NAMES = {
    0: "Normal",
    1: "Surcharge",
    2: "Sous-tension",
    3: "Surtension",
    4: "Surchauffe",
    5: "Défaut sévère",
}

CLASS_ICONS = {
    0: "🟢",
    1: "🟠",
    2: "🟠",
    3: "🔴",
    4: "🟠",
    5: "🔴",
}

# Seuils centralisés (évite les "nombres magiques" dispersés dans le code)
SEUIL_VIGILANCE = 30.0
SEUIL_CRITIQUE = 75.0

# Bornes physiques de simulation
BORNES = {
    "Voltage (V)": (0.0, 600.0),
    "Current (A)": (0.0, 400.0),
    "Temperature (°C)": (-10.0, 130.0),
    "Wind Speed (km/h)": (0.0, 150.0),
}

MAX_HISTORIQUE = 200  # nombre max de points conservés en session

SCENARIOS = {
    "Réseau normal": (220.0, 20.0, 32.0, 15.0),
    "Surcharge": (195.0, 125.0, 68.0, 15.0),
    "Sous-tension": (145.0, 45.0, 48.0, 15.0),
    "Surtension": (330.0, 55.0, 52.0, 20.0),
    "Surchauffe": (215.0, 70.0, 80.0, 8.0),
    "Défaut sévère": (70.0, 260.0, 105.0, 25.0),
}


# ============================================================
# OUTILS MATHÉMATIQUES
# ============================================================
def sigmoid(x: float) -> float:
    """Fonction sigmoïde pour obtenir une évolution progressive (évite les
    changements brusques quand un curseur bouge légèrement)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def niveau_depuis_score(score: float) -> str:
    """Convertit un score de risque (0-100) en niveau qualitatif, à partir
    des seuils centralisés SEUIL_VIGILANCE / SEUIL_CRITIQUE."""
    if score >= SEUIL_CRITIQUE:
        return "CRITIQUE"
    if score >= SEUIL_VIGILANCE:
        return "VIGILANCE"
    return "STABLE"


# ============================================================
# GÉNÉRATION DU DATASET SYNTHÉTIQUE
# ============================================================
@st.cache_data
def generer_dataset(n_par_classe: int = 1200) -> tuple[pd.DataFrame, pd.Series]:
    """
    Génère un dataset synthétique par classe de défaut.
    Les distributions se chevauchent volontairement pour éviter que le
    modèle apprenne uniquement des seuils triviaux.
    """
    rng = np.random.default_rng(42)
    donnees: list[list[float]] = []
    labels: list[int] = []

    gabarits = {
        0: lambda: (rng.normal(220, 7), rng.normal(20, 5), rng.normal(32, 5), rng.normal(15, 6)),
        1: lambda: (rng.normal(195, 25), rng.normal(115, 35), rng.normal(65, 15), rng.normal(15, 8)),
        2: lambda: (rng.normal(145, 35), rng.normal(45, 25), rng.normal(48, 12), rng.normal(15, 8)),
        3: lambda: (rng.normal(330, 65), rng.normal(55, 25), rng.normal(52, 14), rng.normal(20, 10)),
        4: lambda: (rng.normal(215, 15), rng.normal(65, 25), rng.normal(78, 14), rng.normal(8, 6)),
        5: lambda: (
            rng.choice([rng.normal(70, 25), rng.normal(450, 70)]),
            rng.normal(220, 65),
            rng.normal(95, 20),
            rng.normal(25, 15),
        ),
    }

    for classe, generateur in gabarits.items():
        for _ in range(n_par_classe):
            donnees.append(list(generateur()))
            labels.append(classe)

    X = pd.DataFrame(donnees, columns=FEATURES)
    y = pd.Series(labels, name="Défaut")

    for col, (lo, hi) in BORNES.items():
        X[col] = X[col].clip(lo, hi)

    return X, y


# ============================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ============================================================
@st.cache_resource
def entrainer_ia(n_par_classe: int = 1200):
    X, y = generer_dataset(n_par_classe)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    model = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(
        y_test, y_pred,
        labels=list(CLASS_NAMES.keys()),
        target_names=list(CLASS_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        "confusion": confusion_matrix(y_test, y_pred, labels=list(CLASS_NAMES.keys())),
        "report": pd.DataFrame(report).T,
    }
    return model, metrics


# ============================================================
# SCORE DE RISQUE CONTINU (indépendant du modèle ML)
# ============================================================
def calculer_score_risque(tension: float, courant: float, temperature: float, vent: float) -> float:
    """
    Score continu (0-100) basé sur des règles physiques, indépendant du
    modèle ML. Évite les changements brusques quand l'utilisateur déplace
    progressivement les curseurs. C'est une mesure de simulation, pas une
    probabilité physique garantie de panne.
    """
    sous_tension = sigmoid((190 - tension) / 12)
    surtension = sigmoid((tension - 255) / 15)
    surcourant = sigmoid((courant - 70) / 18)
    surchauffe = sigmoid((temperature - 55) / 10)
    vent_extreme = sigmoid((vent - 70) / 18)

    score = (
        0.28 * sous_tension
        + 0.25 * surtension
        + 0.25 * surcourant
        + 0.18 * surchauffe
        + 0.04 * vent_extreme
    )
    return float(np.clip(score * 100, 0, 100))


# ============================================================
# DIAGNOSTIC FUSIONNÉ (règles physiques + avis du modèle ML)
# ============================================================
def diagnostiquer(
    score_risque: float,
    tension: float,
    courant: float,
    temperature: float,
    prediction_ml: int,
) -> dict:
    """
    Diagnostic hybride : les règles physiques (prioritaires, car
    explicables et déterministes) fixent la nature et la sévérité de
    l'anomalie ; le modèle ML sert de second avis. Le dashboard affiche
    explicitement si les deux sont d'accord, ce qui est important pour la
    confiance dans un système de supervision.
    """
    classe_ml = CLASS_NAMES.get(int(prediction_ml), "Anomalie")

    def resultat(nature, niveau, actions, classe_regle):
        accord = CLASS_NAMES.get(classe_regle) == classe_ml if classe_regle is not None else None
        return {
            "nature": nature,
            "niveau": niveau,
            "actions": actions,
            "classe_ml": classe_ml,
            "accord_ml": accord,
        }

    # Défaut sévère : combinaison de plusieurs anomalies extrêmes.
    if tension < 80 or tension > 450 or courant > 280 or temperature > 110:
        return resultat(
            "Défaut sévère / condition électrique anormale",
            "CRITIQUE",
            [
                "Maintenir la zone concernée dans un état sécurisé.",
                "Faire vérifier les protections et les mesures par du personnel habilité.",
                "Identifier la cause avant toute remise en service.",
            ],
            classe_regle=5,
        )

    if tension < 170 and courant > 80:
        niveau = "CRITIQUE" if score_risque >= SEUIL_CRITIQUE else "VIGILANCE"
        return resultat(
            "Sous-tension associée à une forte charge",
            niveau,
            [
                "Vérifier la charge et les protections du départ concerné.",
                "Surveiller l'évolution simultanée de la tension et du courant.",
                "Planifier une inspection si la tendance persiste.",
            ],
            classe_regle=2,
        )

    if tension < 170:
        niveau = "CRITIQUE" if score_risque >= SEUIL_CRITIQUE else "VIGILANCE"
        return resultat(
            "Sous-tension détectée",
            niveau,
            [
                "Contrôler la tension du départ concerné.",
                "Rechercher une surcharge ou une chute de tension.",
                "Comparer avec l'historique des mesures.",
            ],
            classe_regle=2,
        )

    if tension > 280:
        niveau = "CRITIQUE" if score_risque >= SEUIL_CRITIQUE else "VIGILANCE"
        return resultat(
            "Surtension détectée",
            niveau,
            [
                "Vérifier les protections contre les surtensions.",
                "Contrôler les équipements et les mesures de tension.",
                "Ne pas rétablir le fonctionnement sans validation appropriée.",
            ],
            classe_regle=3,
        )

    if courant > 100:
        niveau = "CRITIQUE" if score_risque >= SEUIL_CRITIQUE else "VIGILANCE"
        return resultat(
            "Surcharge / surintensité probable",
            niveau,
            [
                "Contrôler la charge du départ concerné.",
                "Vérifier les protections et les conditions d'exploitation.",
                "Surveiller la température et l'évolution du courant.",
            ],
            classe_regle=1,
        )

    if temperature > 65:
        niveau = "CRITIQUE" if score_risque >= SEUIL_CRITIQUE else "VIGILANCE"
        return resultat(
            "Échauffement anormal",
            niveau,
            [
                "Surveiller la température du conducteur ou de l'équipement.",
                "Vérifier la charge et la ventilation.",
                "Programmer une inspection si l'échauffement persiste.",
            ],
            classe_regle=4,
        )

    # Zone intermédiaire : on s'appuie surtout sur le score continu et le ML.
    niveau = niveau_depuis_score(score_risque)
    if niveau == "STABLE":
        return resultat(
            "Fonctionnement normal",
            "STABLE",
            [
                "Continuer la surveillance normale.",
                "Conserver les mesures pour l'historique.",
            ],
            classe_regle=0,
        )

    return resultat(
        f"Anomalie progressive détectée — indication IA : {classe_ml}",
        niveau,
        [
            "Renforcer la surveillance du réseau.",
            "Comparer les valeurs actuelles aux mesures précédentes.",
            "Programmer un contrôle préventif si la tendance continue.",
        ],
        classe_regle=None,
    )


# ============================================================
# HISTORIQUE RÉEL (accumulé en session, pas simulé à chaque rafraîchissement)
# ============================================================
def init_historique() -> None:
    if "historique" not in st.session_state:
        st.session_state.historique = pd.DataFrame(
            columns=["Temps", *FEATURES, "Risque (%)", "Niveau"]
        )


def ajouter_mesure(tension: float, courant: float, temperature: float, vent: float,
                    score_risque: float, niveau: str) -> None:
    t = len(st.session_state.historique)
    nouvelle_ligne = pd.DataFrame([{
        "Temps": t,
        "Voltage (V)": tension,
        "Current (A)": courant,
        "Temperature (°C)": temperature,
        "Wind Speed (km/h)": vent,
        "Risque (%)": score_risque,
        "Niveau": niveau,
    }])
    st.session_state.historique = pd.concat(
        [st.session_state.historique, nouvelle_ligne], ignore_index=True
    ).tail(MAX_HISTORIQUE)


# ============================================================
# APPLICATION
# ============================================================
st.title("⚡ Système intelligent de supervision et détection des défauts")
st.caption(
    "Prototype de recherche : apprentissage automatique + score de risque "
    "continu + historique réel de simulation."
)

init_historique()

# -------------------- SIDEBAR --------------------
st.sidebar.header("🎛️ Centre de simulation")

mode = st.sidebar.selectbox("Mode de simulation", ["Personnalisé", *SCENARIOS.keys()])

if mode != "Personnalisé":
    default_v, default_i, default_t, default_w = SCENARIOS[mode]
else:
    default_v, default_i, default_t, default_w = 220.0, 20.0, 32.0, 15.0

val_tension = st.sidebar.slider(
    "⚡ Tension (V)", *BORNES["Voltage (V)"], float(default_v), 1.0,
    help="Zone nominale de démonstration : environ 220 V.",
)
val_courant = st.sidebar.slider("🔌 Courant (A)", *BORNES["Current (A)"], float(default_i), 1.0)
val_temp = st.sidebar.slider("🌡️ Température (°C)", *BORNES["Temperature (°C)"], float(default_t), 0.5)
val_vent = st.sidebar.slider("💨 Vent (km/h)", *BORNES["Wind Speed (km/h)"], float(default_w), 1.0)

with st.sidebar.expander("⚙️ Options avancées"):
    n_par_classe = st.slider(
        "Taille du dataset d'entraînement (par classe)", 300, 3000, 1200, 100,
        help="Plus grand = entraînement plus long mais potentiellement plus stable.",
    )
    if st.button("🔄 Vider l'historique de session"):
        st.session_state.historique = st.session_state.historique.iloc[0:0]
        st.rerun()

# -------------------- MODÈLE ET PRÉDICTION --------------------
model, metrics = entrainer_ia(n_par_classe)

donnees_utilisateur = pd.DataFrame([{
    "Voltage (V)": val_tension,
    "Current (A)": val_courant,
    "Temperature (°C)": val_temp,
    "Wind Speed (km/h)": val_vent,
}])

probabilites_ml = model.predict_proba(donnees_utilisateur[FEATURES])[0]
prediction_ml = int(model.predict(donnees_utilisateur[FEATURES])[0])
confiance_ml = float(np.max(probabilites_ml) * 100)

score_risque = calculer_score_risque(val_tension, val_courant, val_temp, val_vent)
diagnostic = diagnostiquer(score_risque, val_tension, val_courant, val_temp, prediction_ml)
niveau = diagnostic["niveau"]

ajouter_mesure(val_tension, val_courant, val_temp, val_vent, score_risque, niveau)

# -------------------- EN-TÊTE D'ÉTAT --------------------
if niveau == "CRITIQUE":
    st.error(f"🔴 ÉTAT CRITIQUE — Score de risque : {score_risque:.1f}/100")
elif niveau == "VIGILANCE":
    st.warning(f"🟠 VIGILANCE — Score de risque : {score_risque:.1f}/100")
else:
    st.success(f"🟢 RÉSEAU STABLE — Score de risque : {score_risque:.1f}/100")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tension", f"{val_tension:.1f} V")
c2.metric("Courant", f"{val_courant:.1f} A")
c3.metric("Température", f"{val_temp:.1f} °C")
c4.metric("Risque", f"{score_risque:.1f} / 100")
c5.metric("Confiance IA", f"{confiance_ml:.1f} %")

st.markdown("---")

# -------------------- ONGLETS --------------------
tab_diag, tab_tendances, tab_perf, tab_hist, tab_info = st.tabs(
    ["🔮 Diagnostic", "📈 Tendances", "📊 Performance IA", "📋 Historique", "ℹ️ À propos"]
)

with tab_diag:
    col1, col2 = st.columns(2)

    with col1:
        icon = "🔴" if niveau == "CRITIQUE" else "🟠" if niveau == "VIGILANCE" else "🟢"
        st.subheader(f"{icon} {diagnostic['nature']}")
        st.write(f"**Niveau :** {niveau}")
        st.write(f"**Classe prédite par l'IA :** {CLASS_ICONS[prediction_ml]} {diagnostic['classe_ml']}")
        st.write(f"**Confiance de la classe IA :** {confiance_ml:.1f} %")

        if diagnostic["accord_ml"] is True:
            st.info("✅ Le diagnostic par règles physiques et la classe prédite par l'IA concordent.")
        elif diagnostic["accord_ml"] is False:
            st.warning(
                "⚠️ Désaccord entre les règles physiques et le modèle IA — "
                "à interpréter avec prudence, une vérification manuelle est recommandée."
            )

        st.subheader("🛠️ Recommandations")
        for i, action in enumerate(diagnostic["actions"], start=1):
            st.write(f"**{i}.** {action}")

    with col2:
        st.subheader("🧠 Répartition des probabilités du modèle")
        proba_df = pd.DataFrame({
            "Défaut": [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
            "Probabilité (%)": probabilites_ml * 100,
        }).sort_values("Probabilité (%)", ascending=False)
        st.dataframe(
            proba_df.style.format({"Probabilité (%)": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

        st.subheader("🔍 Importance des variables")
        importance_df = pd.DataFrame({
            "Variable": FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(importance_df["Variable"][::-1], importance_df["Importance"][::-1])
        ax.set_xlabel("Importance relative")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab_tendances:
    hist = st.session_state.historique
    if len(hist) < 2:
        st.info("Déplace les curseurs pour accumuler des points d'historique et voir les tendances.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Évolution des mesures")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(hist["Temps"], hist["Voltage (V)"], label="Tension (V)")
            ax1.plot(hist["Temps"], hist["Current (A)"], label="Courant (A)")
            ax1.plot(hist["Temps"], hist["Temperature (°C)"], label="Température (°C)")
            ax1.set_xlabel("Interaction n°")
            ax1.set_ylabel("Valeur")
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with g2:
            st.subheader("Évolution du risque")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(hist["Temps"], hist["Risque (%)"], linewidth=2)
            ax2.axhline(SEUIL_VIGILANCE, linestyle="--", color="orange", label="Seuil vigilance")
            ax2.axhline(SEUIL_CRITIQUE, linestyle="--", color="red", label="Seuil critique")
            ax2.set_ylim(0, 100)
            ax2.set_xlabel("Interaction n°")
            ax2.set_ylabel("Score de risque")
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

with tab_perf:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f} %")
    m2.metric("Precision", f"{metrics['precision'] * 100:.2f} %")
    m3.metric("Recall", f"{metrics['recall'] * 100:.2f} %")
    m4.metric("F1-score", f"{metrics['f1'] * 100:.2f} %")
    m5.metric("ROC-AUC (macro)", f"{metrics['roc_auc_ovr'] * 100:.2f} %")

    st.write("### Rapport de classification détaillé")
    st.dataframe(metrics["report"].round(3), use_container_width=True)

    st.write("### Matrice de confusion")
    cm = metrics["confusion"]
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    image = ax4.imshow(cm)
    ax4.set_xticks(range(len(CLASS_NAMES)))
    ax4.set_yticks(range(len(CLASS_NAMES)))
    ax4.set_xticklabels(CLASS_NAMES.values(), rotation=35, ha="right")
    ax4.set_yticklabels(CLASS_NAMES.values())
    ax4.set_xlabel("Classe prédite")
    ax4.set_ylabel("Classe réelle")
    ax4.set_title("Matrice de confusion")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, cm[i, j], ha="center", va="center")
    fig4.colorbar(image, ax=ax4)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    st.caption(
        "Ces métriques mesurent la capacité du modèle à distinguer les distributions "
        "synthétiques définies dans le code — pas une performance sur données réelles."
    )

with tab_hist:
    st.dataframe(st.session_state.historique.round(2), use_container_width=True, hide_index=True)
    csv = st.session_state.historique.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger l'historique CSV", data=csv,
        file_name="historique_simulation_reseau.csv", mime="text/csv",
    )

with tab_info:
    st.markdown(
        """
        **Chaîne de fonctionnement :**

        1. Acquisition des valeurs simulées (sliders ou scénarios prédéfinis).
        2. Vérification et limitation des valeurs physiques de simulation.
        3. Calcul d'un score de risque continu (règles physiques, indépendant du ML).
        4. Classification multi-classes avec XGBoost.
        5. Fusion du diagnostic par règles et de la prédiction IA, avec
           indicateur explicite d'accord/désaccord entre les deux.
        6. Accumulation d'un historique réel en session (pas simulé à chaque
           rafraîchissement).
        7. Visualisation des tendances et export CSV.
        8. Évaluation du modèle : accuracy, precision, recall, F1, ROC-AUC,
           rapport de classification et matrice de confusion.

        **Important :** les données utilisées ici sont synthétiques.
        Ce prototype doit être validé sur des données électriques réelles
        avant toute utilisation industrielle.
        """
    )

st.caption(
    "⚠️ Prototype de simulation — ne pas utiliser comme système de "
    "protection ou de commande d'une installation électrique réelle."
)
