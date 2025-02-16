import streamlit as st
import requests
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Défaut de Crédit", page_icon="💰", layout="wide"
)

# Style personnalisé
st.markdown(
    """
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Titre de l'application
st.title("Système de Scoring Crédit")
st.markdown("### Prédiction de la probabilité de défaut de paiement")

# Sidebar avec les informations
with st.sidebar:
    st.header("Instructions")
    st.write("1. Entrez l'ID du client")
    st.write("2. Cliquez sur 'Prédire'")
    st.write("3. Consultez les résultats")

    st.markdown("---")
    st.markdown("### À propos")
    st.write(
        "Ce système aide à la décision d'octroi de crédit en prédisant la probabilité de défaut de paiement d'un client."
    )

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    # Input pour l'ID client
    client_id = st.number_input(
        "ID Client",
        min_value=100002,  # remplacer par MIN_ID
        max_value=456255,  # remplacer par MAX_ID
        step=1,
    )

    if st.button("Prédire"):
        try:
            # Appel à l'API
            response = requests.get(f"http://127.0.0.1:8000/predict/{client_id}")

            if response.status_code == 200:
                result = response.json()

                # Affichage des résultats
                st.markdown("---")
                st.markdown("### Résultats de l'analyse")

                # Probabilité de défaut
                st.markdown("#### Probabilité de défaut")
                proba = result["probability"]
                st.progress(proba)
                st.write(f"Probabilité: {proba:.1%}")

                # Décision
                st.markdown("#### Décision")
                if result["prediction"] == 1:
                    st.error(f"🚫 {result['decision']}")
                else:
                    st.success(f"✅ {result['decision']}")

            elif response.status_code == 400:
                st.error(response.json()["detail"])
            else:
                st.error("Erreur lors de la récupération des données")

        except Exception as e:
            st.error(f"Erreur lors de la connexion à l'API: {str(e)}")

with col2:
    st.markdown("### Seuil de décision")
    st.write("Seuil utilisé : 36%")
    st.write(
        "Un client avec une probabilité de défaut supérieure à 36% se verra refuser le crédit."
    )

# Footer
st.markdown("---")
st.markdown("Projet 7 - OpenClassrooms Data Scientist")
