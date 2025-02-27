import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import shap
from PIL import Image
import os
import time

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Défaut de Crédit", page_icon="💰", layout="wide"
)

# Variables d'environnement
API_URL = "https://projet-7-docker.onrender.com"

# Dictionnaire de descriptions des caractéristiques
feature_descriptions = {
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "DAYS_EMPLOYED": "Jours d'emploi",
    "ACTIVE_DAYS_CREDIT_MAX": "Jours max crédit actif",
    "PREV_CNT_PAYMENT_MEAN": "Nombre moyen de paiements",
    "PAYMENT_RATE": "Taux de paiement",
    "INSTAL_DBD_SUM": "Jours en retard de paiement",
    "INSTAL_DPD_MEAN": "Jours en retard moyen",
    "BURO_DAYS_CREDIT_MAX": "Jours max crédit bureau",
    "PREV_NAME_CONTRACT_STATUS_Refused_MEAN": "Taux de refus précédent",
    "NAME_EDUCATION_TYPE_Higher_education": "Niveau d'études supérieur",
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN": "Dette moyenne crédit bureau",
    "AMT_ANNUITY": "Montant annuité",
    "POS_SK_DPD_DEF_MEAN": "Jours moyen retard points de vente",
    "BURO_DAYS_CREDIT_ENDDATE_MAX": "Date fin max crédit bureau",
    "CODE_GENDER": "Genre",
    "BURO_DAYS_CREDIT_MEAN": "Jours moyen crédit bureau",
    "APPROVED_AMT_DOWN_PAYMENT_MAX": "Acompte max approuvé",
    "INSTAL_PAYMENT_DIFF_MEAN": "Différence moyenne de paiement",
    "DAYS_BIRTH": "Âge en jours",
}


# Chargement du modèle et des données pour l'analyse locale
@st.cache_resource
def load_model_and_data():
    """Charge le modèle et les données nécessaires pour l'analyse locale"""
    try:
        # Construire les chemins relatifs corrects
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "median_model.joblib")
        index_path = os.path.join(base_dir, "client_index.pkl")
        data_path = os.path.join(base_dir, "DF_median_impute.csv")

        # Charger les fichiers avec les chemins corrects
        model = joblib.load(model_path)

        with open(index_path, "rb") as f:
            headers, client_index = pickle.load(f)

        # Charger un échantillon de données pour les distributions (limité pour améliorer les performances)
        df_sample = pd.read_csv(data_path, index_col="SK_ID_CURR", nrows=3000)

        # Charger/calculer les features les plus importantes (top 10)
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": model.feature_names_in_,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )

        return model, headers, client_index, df_sample, feature_importance
    except Exception as e:
        st.error(f"Erreur lors du chargement des données locales: {str(e)}")
        # Afficher des informations de débogage sur les chemins
        st.error(f"Vérifiez les chemins: Base={base_dir}, Model={model_path}")
        return None, None, None, None, None


# Fonction pour obtenir les données d'un client
@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def get_client_data(client_id):
    """Récupère les données d'un client depuis l'API"""
    try:
        response = requests.get(f"{API_URL}/client-data/{client_id}")
        if response.status_code == 200:
            return response.json()["data"]
        else:
            st.error(f"Erreur lors de la récupération des données: {response.json()}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion: {str(e)}")
        return None


# Fonction pour faire une prédiction
@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def predict_client(client_id):
    """Effectue une prédiction pour un client via l'API"""
    try:
        response = requests.get(f"{API_URL}/predict/{client_id}")
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(
                f"Client non trouvé: {response.json().get('detail', {}).get('message')}"
            )
            return None
        else:
            st.error(f"Erreur: {response.json()}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None


# Calcul des valeurs SHAP une seule fois
@st.cache_data
def calculate_shap_values(_model, client_data, feature_names):
    """Calcule les valeurs SHAP une seule fois et les met en cache"""
    client_df = pd.DataFrame([client_data], columns=feature_names)
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(client_df)

    # Traitement du format des valeurs SHAP
    if isinstance(shap_values, list) and len(shap_values) > 1:
        values = shap_values[1]
    else:
        values = shap_values

    return values, client_df


# Obtenir les caractéristiques importantes pour un client spécifique
def get_client_important_features(values, feature_names, num_features=5):
    """Version optimisée qui utilise des valeurs SHAP pré-calculées"""
    # Créer un DataFrame avec les valeurs SHAP absolues
    shap_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(values[0])})

    # Trier par importance et retourner les top features
    return (
        shap_df.sort_values("importance", ascending=False)
        .head(num_features)["feature"]
        .tolist()
    )


# Fonction pour générer l'explication SHAP locale
def generate_improved_shap_explanation(
    values, client_df, feature_names, num_features=10
):
    """Version optimisée qui utilise des valeurs SHAP pré-calculées"""
    # Créer un DataFrame avec les valeurs SHAP et les valeurs des caractéristiques
    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP_Value": values[0],
            "Feature_Value": client_df.iloc[0].values,
            "Abs_Impact": np.abs(values[0]),
        }
    )

    # Trier par impact absolu et prendre les top caractéristiques
    shap_df = shap_df.sort_values("Abs_Impact", ascending=False).head(num_features)

    # Appliquer les descriptions améliorées si disponibles
    shap_df["Feature_Display"] = shap_df["Feature"].map(
        lambda x: feature_descriptions.get(x, x)
    )

    # Créer une échelle de couleurs accessible
    colors = []
    for val in shap_df["SHAP_Value"]:
        if val < 0:  # Impact négatif (réduit le risque)
            colors.append("#1E88E5")  # Bleu
        else:  # Impact positif (augmente le risque)
            colors.append("#F57C00")  # Orange

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(18, 14))

    # Barres horizontales
    bars = ax.barh(
        y=shap_df["Feature_Display"],
        width=shap_df["SHAP_Value"],
        color=colors,
        height=0.7,
    )

    # Ajouter une ligne verticale à zéro
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    # Augmenter la taille de la police pour l'axe Y (noms des caractéristiques)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="x", labelsize=20)

    # Annotation de chaque barre avec la valeur de la caractéristique
    for i, bar in enumerate(bars):
        feature_value = shap_df.iloc[i]["Feature_Value"]
        # Formater les valeurs pour l'affichage
        if isinstance(feature_value, (int, float)):
            if abs(feature_value) >= 1000:
                value_text = f"{feature_value:.0f}"
            elif abs(feature_value) >= 100:
                value_text = f"{feature_value:.1f}"
            else:
                value_text = f"{feature_value:.2f}"
        else:
            value_text = str(feature_value)

        # Positionner le texte
        x_pos = bar.get_width()
        if x_pos < 0:
            x_pos = x_pos - 0.01
            ha = "right"
        else:
            x_pos = x_pos + 0.01
            ha = "left"

        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            value_text,
            va="center",
            ha=ha,
            fontweight="bold",
            color="black",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    # Styliser le graphique
    ax.set_title(
        "Impact des caractéristiques sur le risque de défaut",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xlabel("Impact sur la prédiction", fontsize=16)

    # Augmenter l'espacement entre les barres
    ax.set_ylim([-0.5, len(shap_df) - 0.5])

    # Ajouter une marge à gauche pour les noms longs
    plt.subplots_adjust(left=0.3)

    # Ajouter une note explicative avec une police plus grande
    ax.text(
        0,
        -0.6,
        "Note: Les barres à droite (orange) indiquent les facteurs qui augmentent le risque de défaut de paiement.\n"
        "Les barres à gauche (bleues) indiquent les facteurs qui réduisent ce risque.\n"
        "Les valeurs à côté de chaque barre montrent la valeur réelle de la caractéristique pour ce client.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=24,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


# Pré-calculer les histogrammes pour les features importantes
@st.cache_data
def precalculate_histograms(df_sample, features_list, bins=30):
    """Pré-calcule les histogrammes pour un ensemble de features"""
    histograms = {}
    for feat in features_list:
        if feat in df_sample.columns:
            # Extraire la série et enlever les NaN
            series = df_sample[feat].dropna()

            # Calculer les bins et les fréquences une fois
            hist, bin_edges = np.histogram(series, bins=bins)

            # Au lieu de stocker une fonction lambda, stockez les données nécessaires
            # pour calculer le percentile plus tard
            sorted_values = np.sort(series.values)

            # Calculer quelques statistiques utiles
            histograms[feat] = {
                "hist": hist,
                "bin_edges": bin_edges,
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
                "sorted_values": sorted_values,  # Pour calculer le percentile plus tard
                "count": len(sorted_values),
            }
    return histograms


# Fonction optimisée pour afficher la distribution d'une variable
def plot_feature_distribution_optimized(histograms, client_value, feature_name):
    """Version optimisée qui utilise des histogrammes pré-calculés"""
    if feature_name not in histograms:
        return None

    # Obtenir un nom plus descriptif pour l'affichage
    display_name = feature_descriptions.get(feature_name, feature_name)

    # Utiliser les données pré-calculées
    hist_data = histograms[feature_name]

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 4))

    # Tracer l'histogramme manuellement
    width = np.diff(hist_data["bin_edges"])
    centers = (hist_data["bin_edges"][:-1] + hist_data["bin_edges"][1:]) / 2

    ax.bar(centers, hist_data["hist"], width=width, alpha=0.6, color="#4285F4")

    # Ajouter la courbe KDE (approximation rapide)
    x = np.linspace(hist_data["min"], hist_data["max"], 100)
    # Créer une approximation de la KDE en utilisant une moyenne mobile des bins
    kde_y = np.interp(
        x, centers, hist_data["hist"] / sum(hist_data["hist"]) * len(centers)
    )
    kde_y = np.convolve(kde_y, np.ones(5) / 5, mode="same")
    kde_y = kde_y / np.max(kde_y) * np.max(hist_data["hist"])
    ax.plot(x, kde_y, "r-", linewidth=1.5)

    # Ajouter une ligne pour la valeur du client
    ax.axvline(x=client_value, color="red", linestyle="--", linewidth=2)

    # Calculer le percentile du client en utilisant les valeurs triées
    sorted_values = hist_data["sorted_values"]
    count = hist_data["count"]
    # Trouver l'indice le plus proche
    idx = np.searchsorted(sorted_values, client_value)
    # Calculer le percentile
    client_percentile = (idx / count) * 100

    # Ajouter du texte pour indiquer la position du client
    ax.text(
        client_value,
        ax.get_ylim()[1] * 0.9,
        f"Client: {client_value:.2f}\nPercentile: {client_percentile:.1f}%",
        horizontalalignment="left" if client_percentile < 50 else "right",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.title(f"Distribution de {display_name}", fontsize=14)
    plt.tight_layout()
    return fig


# Style personnalisé pour l'accessibilité
st.markdown(
    """
    <style>
    /* Texte plus grand pour la lisibilité */
    body {
        font-size: 18px;
    }
    
    /* En-têtes plus grands et contrastés */
    h1, h2, h3 {
        color: #1E3A8A;
    }
    
    /* Style pour les grands textes */
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    
    /* Style pour les informations critiques */
    .important-info {
        font-size: 20px;
        padding: 10px;
        border-radius: 5px;
        background-color: #F8F9FA;
        border-left: 5px solid #1E88E5;
    }
    
    /* Amélioration des contrastes pour les éléments de décision */
    .decision-accept {
        background-color: #D5F5E3;
        color: #145A32;
        padding: 15px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }
    
    .decision-reject {
        background-color: #FADBD8;
        color: #943126;
        padding: 15px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    /* Animation pendant le chargement */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .loading-pulse {
        animation: pulse 1.5s infinite;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Charger le modèle et les données en cache
model, headers, client_index, df_sample, top_features = load_model_and_data()

# Titre de l'application
st.title("Système de Scoring Crédit")
st.markdown("### Prédiction de la probabilité de défaut de paiement")

# Sidebar avec les informations
with st.sidebar:
    st.header("Instructions")
    st.write("1. Entrez l'ID du client")
    st.write("2. Cliquez sur 'Prédire'")
    st.write("3. Consultez les résultats et explications")

    st.markdown("---")

    st.markdown("### À propos")
    st.write(
        "Ce système aide à la décision d'octroi de crédit en prédisant la probabilité de défaut de paiement d'un client."
    )

    st.markdown("---")

    # Ajout des paramètres d'affichage
    st.header("Paramètres d'affichage")
    show_explanations = st.checkbox("Afficher les explications détaillées", value=True)
    num_features_to_show = st.slider(
        "Nombre de caractéristiques à afficher", min_value=3, max_value=10, value=3
    )

    # Mode performance
    performance_mode = st.checkbox(
        "Mode performance (chargement plus rapide)", value=True
    )
    num_bins = 10 if performance_mode else 50  # Moins de bins = plus rapide

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    # Input pour l'ID client
    # Récupérer les ID min et max des clients depuis l'API ou localement
    min_id = 100002  # Valeur par défaut
    max_id = 456255  # Valeur par défaut

    if client_index:
        min_id = min(client_index.keys())
        max_id = max(client_index.keys())

    client_id = st.number_input(
        "ID Client",
        min_value=min_id,
        max_value=max_id,
        step=1,
        help="Entrez l'identifiant unique du client pour obtenir une prédiction",
    )

    predict_button = st.button("Prédire", type="primary")

    # Placeholder pour les résultats
    result_placeholder = st.empty()
    explanation_placeholder = st.empty()
    distributions_placeholder = st.empty()

    if predict_button:
        # Affichage d'un indicateur de chargement
        with result_placeholder.container():
            st.markdown(
                """
                <div class="loading-pulse" style="text-align: center; padding: 20px;">
                    <h3>Chargement de la prédiction...</h3>
                    <p>Récupération des données du client et calcul du score de risque</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Récupérer les résultats en arrière-plan
            start_time = time.time()
            result = predict_client(client_id)

            if not result:
                result_placeholder.error(
                    "Impossible d'obtenir les résultats pour ce client"
                )
            else:
                # Effacer le message de chargement
                result_placeholder.empty()

                # Récupérer les informations de prédiction
                prediction_info = result["prediction"]
                proba = prediction_info["probability"]
                decision = prediction_info["decision"]

                # Affichage des résultats
                with result_placeholder.container():
                    st.markdown("### Résultats de l'analyse")

                    # Probabilité de défaut avec barre de progression colorée
                    st.markdown("#### Probabilité de défaut")

                    # Couleur conditionnelle pour la barre de progression
                    color = "#ff6b6b" if proba >= 0.34 else "#63c132"
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 10px; font-weight: bold;">
                            Probabilité: {proba:.1%}
                        </div>
                        <div style="height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="width: {proba*100}%; height: 100%; background-color: {color};"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Seuil
                    st.markdown(
                        f"""
                        <div style="text-align: right; font-style: italic; margin-top: 5px;">
                            Seuil de décision: 34%
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Décision avec style amélioré
                    st.markdown("#### Décision")
                    if decision == "Refusé":
                        st.markdown(
                            f"""
                            <div class="decision-reject">
                                🚫 {decision}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="decision-accept">
                                ✅ {decision}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Si les explications sont activées, les préparer en arrière-plan
                if show_explanations:
                    with explanation_placeholder.container():
                        st.markdown(
                            """
                            <div class="loading-pulse" style="text-align: center; padding: 20px;">
                                <h3>Génération des explications...</h3>
                                <p>Calcul des facteurs qui ont influencé la décision</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Récupérer les données client
                        client_data = get_client_data(client_id)

                        if client_data and model:
                            # Calculer les valeurs SHAP une seule fois
                            shap_values, client_df = calculate_shap_values(
                                model, client_data, model.feature_names_in_
                            )

                            # Obtenir les caractéristiques importantes pour ce client
                            client_important_features = get_client_important_features(
                                shap_values,
                                model.feature_names_in_,
                                num_features_to_show,
                            )

                            # Pré-calculer les histogrammes pour ces caractéristiques
                            all_important_features = client_important_features
                            histograms = precalculate_histograms(
                                df_sample, all_important_features, bins=num_bins
                            )

                            # Effacer le message de chargement
                            explanation_placeholder.empty()

                            # Afficher les explications
                            with explanation_placeholder.container():
                                st.markdown("### Explication de la décision")

                                # Afficher le graphique SHAP
                                shap_fig = generate_improved_shap_explanation(
                                    shap_values,
                                    client_df,
                                    model.feature_names_in_,
                                    num_features=8,
                                )
                                st.pyplot(shap_fig, use_container_width=True)

                            # Afficher les distributions pour les top features du client
                            # Afficher les distributions pour les top features du client
                            with distributions_placeholder.container():
                                st.markdown(
                                    "### Position du client sur ses caractéristiques clés"
                                )

                                # Afficher chaque distribution individuellement pour maximiser la taille
                                for feat in client_important_features:
                                    if feat in client_data and feat in histograms:
                                        st.subheader(
                                            f"Distribution de {feature_descriptions.get(feat, feat)}"
                                        )
                                        client_value = client_data[feat]
                                        dist_fig = plot_feature_distribution_optimized(
                                            histograms, client_value, feat
                                        )
                                        if dist_fig:
                                            st.pyplot(
                                                dist_fig, use_container_width=True
                                            )  # Utiliser toute la largeur
                                            plt.close(dist_fig)  # Nettoyer la mémoire

                            # Afficher le temps de traitement
                            end_time = time.time()
                            st.caption(
                                f"Temps de traitement total: {end_time - start_time:.2f} secondes"
                            )
                        else:
                            explanation_placeholder.error(
                                "Impossible de générer les explications pour ce client"
                            )
                else:
                    # Si les explications sont désactivées, simplement noter le temps
                    end_time = time.time()
                    st.caption(
                        f"Temps de traitement: {end_time - start_time:.2f} secondes"
                    )

with col2:
    st.markdown("### Seuil de décision")
    st.markdown(
        """
        <div style="background-color: #F8F9FA; color: #1E3A8A; padding: 10px; 
        border-radius: 5px; border-left: 5px solid #1E88E5; font-size: 18px;">
            Un client avec une probabilité de défaut supérieure ou égale à <strong>34%</strong> 
            se verra refuser le crédit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Comment interpréter les résultats")
    st.write(
        """
    - La **probabilité de défaut** indique le risque que le client ne rembourse pas son crédit.
    - Les **explications** montrent quelles caractéristiques ont influencé la décision.
    - Les **distributions** montrent où se situe le client par rapport aux autres.
    """
    )

    # Informations sur les caractéristiques importantes
    if top_features is not None:
        st.markdown("### Caractéristiques les plus importantes")
        for i, (feature, importance) in enumerate(
            zip(top_features["feature"].head(5), top_features["importance"].head(5))
        ):
            feature_name = feature_descriptions.get(feature, feature)
            st.write(f"{i+1}. **{feature_name}** ")

# Footer
st.markdown("---")
st.markdown("Projet 7 - OpenClassrooms Data Scientist")
st.markdown(
    "<div style='text-align: center; color: #666;'>Système d'aide à la décision pour l'octroi de crédit</div>",
    unsafe_allow_html=True,
)
