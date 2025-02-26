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

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Défaut de Crédit", page_icon="💰", layout="wide"
)

# Variables d'environnement
API_URL = "https://projet-7-docker.onrender.com"  # URL configurable


# Chargement du modèle et des données pour l'analyse locale
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

        # Charger un échantillon de données pour les distributions
        df_sample = pd.read_csv(data_path, index_col="SK_ID_CURR", nrows=5000)

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


# Fonction pour générer l'explication SHAP locale
def generate_improved_shap_explanation(
    model, client_data, feature_names, num_features=10
):
    """Génère une explication SHAP améliorée et accessible pour un client spécifique"""
    # Préparer les données
    client_df = pd.DataFrame([client_data], columns=feature_names)

    # Créer l'explainer SHAP
    explainer = shap.TreeExplainer(model)

    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(client_df)

    # Vérifier le format des valeurs SHAP
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Format pour classification binaire [classe0, classe1]
        values = shap_values[1]
    else:
        # Format unique array
        values = shap_values

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

    # Transformer les noms de caractéristiques avec des sauts de ligne pour les longs noms
    feature_descriptions = {
        "EXT_SOURCE_1": "Score externe 1",
        "EXT_SOURCE_2": "Score externe 2",
        "EXT_SOURCE_3": "Score externe 3",
        "DAYS_EMPLOYED": "Jours d'emploi",
        "ACTIVE_DAYS_CREDIT_MAX": "Jours max\ncrédit actif",
        "PREV_CNT_PAYMENT_MEAN": "Nombre moyen\nde paiements",
        "PAYMENT_RATE": "Taux de paiement",
        "INSTAL_DBD_SUM": "Jours en retard\nde paiement",
        "INSTAL_DPD_MEAN": "Jours en retard\nmoyen",
        "BURO_DAYS_CREDIT_MAX": "Jours max\ncrédit bureau",
        "PREV_NAME_CONTRACT_STATUS_Refused_MEAN": "Taux de refus\nprécédent",
        "NAME_EDUCATION_TYPE_Higher_education": "Niveau d'études\nsupérieur",
        "BURO_AMT_CREDIT_SUM_DEBT_MEAN": "Dette moyenne\ncrédit bureau",
        "AMT_ANNUITY": "Montant annuité",
        "POS_SK_DPD_DEF_MEAN": "Jours moyen retard\npoints de vente",
        "BURO_DAYS_CREDIT_ENDDATE_MAX": "Date fin max\ncrédit bureau",
        "CODE_GENDER": "Genre",
        "BURO_DAYS_CREDIT_MEAN": "Jours moyen\ncrédit bureau",
        "APPROVED_AMT_DOWN_PAYMENT_MAX": "Acompte max\napprouvé",
        "INSTAL_PAYMENT_DIFF_MEAN": "Différence moyenne\nde paiement",
        # Ajoutez d'autres descriptions selon vos besoins
    }

    # Appliquer les descriptions améliorées si disponibles
    shap_df["Feature_Display"] = shap_df["Feature"].map(
        lambda x: feature_descriptions.get(x, x)
    )

    # Créer une échelle de couleurs accessible
    # Utiliser une échelle bleu-orange (meilleure pour le daltonisme)
    colors = []
    for val in shap_df["SHAP_Value"]:
        if val < 0:  # Impact négatif (réduit le risque)
            colors.append("#1E88E5")  # Bleu
        else:  # Impact positif (augmente le risque)
            colors.append("#F57C00")  # Orange

    # Augmenter la taille du graphique
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
        fontsize=18,
        style="italic",
    )

    plt.tight_layout()
    return fig


# Fonction pour afficher la distribution d'une variable avec la position du client
def plot_feature_distribution(df, client_value, feature_name):
    """Affiche la distribution d'une caractéristique avec la position du client"""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Tracer la distribution
    sns.histplot(df[feature_name].dropna(), kde=True, ax=ax)

    # Ajouter une ligne pour la valeur du client
    plt.axvline(x=client_value, color="red", linestyle="--", linewidth=2)

    # Ajouter du texte pour indiquer la position du client
    client_percentile = (
        sum(df[feature_name] <= client_value) / len(df[feature_name].dropna()) * 100
    )
    plt.text(
        client_value,
        ax.get_ylim()[1] * 0.9,
        f"Client: {client_value:.2f}\nPercentile: {client_percentile:.1f}%",
        horizontalalignment="left" if client_percentile < 50 else "right",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.title(f"Distribution de {feature_name}", fontsize=14)
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
        "Nombre de caractéristiques à afficher", min_value=3, max_value=10, value=5
    )

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

    if predict_button:
        # Affichage d'un indicateur de chargement
        with st.spinner("Analyse en cours..."):
            result = predict_client(client_id)

            if result:
                # Récupérer les informations de prédiction
                prediction_info = result["prediction"]
                proba = prediction_info["probability"]
                decision = prediction_info["decision"]

                # Récupérer les données client si nécessaire pour les explications
                if show_explanations:
                    client_data = get_client_data(client_id)

                # Affichage des résultats
                st.markdown("---")
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

                # Explications
                if show_explanations and client_data and model:
                    st.markdown("---")
                    st.markdown("### Explication de la décision")

                    # Afficher l'importance globale des features pour ce client
                    shap_fig = generate_improved_shap_explanation(
                        model, client_data, model.feature_names_in_, num_features=8
                    )
                    st.pyplot(shap_fig, use_container_width=True)

                    # Afficher les distributions pour les top features
                    st.markdown("### Position du client sur les caractéristiques clés")

                    # Sélectionner les top features à afficher
                    features_to_show = top_features["feature"].tolist()[
                        :num_features_to_show
                    ]

                    for feat in features_to_show:
                        if feat in client_data and feat in df_sample.columns:
                            client_value = client_data[feat]
                            dist_fig = plot_feature_distribution(
                                df_sample, client_value, feat
                            )
                            st.pyplot(dist_fig)
                            plt.close(dist_fig)  # Nettoyer la mémoire

with col2:
    st.markdown("### Seuil de décision")
    st.markdown(
        """
        <div style="background-color: #F8F9FA; color: #1E3A8A; padding: 10px; 
        border-radius: 5px; border-left: 5px solid #1E88E5; font-size: 18px;">
            Un client avec une probabilité de défaut supérieure à <strong>34%</strong> se verra refuser le crédit.
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
            st.write(f"{i+1}. **{feature}** - {importance:.4f}")

# Footer
st.markdown("---")
st.markdown("Projet 7 - OpenClassrooms Data Scientist")
st.markdown(
    "<div style='text-align: center; color: #666;'>Système d'aide à la décision pour l'octroi de crédit</div>",
    unsafe_allow_html=True,
)
