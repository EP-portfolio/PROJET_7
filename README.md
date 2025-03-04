Projet de Scoring Crédit - Prédiction des défauts de paiement

Présentation du projet

Ce projet implémente un système de scoring crédit qui évalue la probabilité de défaut de paiement d'un client. Il s'inscrit dans une démarche d'automatisation 
de la décision d'octroi de crédit avec un accent sur l'explicabilité des décisions.

Le système comprend :

•	Un modèle de prédiction (LightGBM) optimisé ainsi que d’autres modèles testés

•	Une API de prédiction déployée dans le cloud

•	Un dashboard interactif pour visualiser les décisions et leur explication

•	Un système de monitoring pour détecter le data drift
 

Structure du projet

|-- api/                             # Code source de l'API

    |-- api.py                         # Implémentation FastAPI
  
|--models/                           # Modèles sauvegardés

    |-- median_model.joblib              # Modèle LightGBM (imputation médiane 47 features, hyperparamétres optimisés, modèle final)

    |-- 0_model.joblib                   # Modèle LightGBM (imputation 0)

    |-- KNN_model.joblib                 # Modèle LightGBM (imputation KNN)

    |-- credit_model.joblib              # Modèle régression logistique (imputation moyenne)

    |-- lgbm_model.joblib                # Modèle LightGBM (imputation moyenne, 700+ features)

    |-- lgbm_optimized_model.joblib      # Modèle LightGBM (imputation moyenne, 700+ features, hyperparamétres optimisés)

    |-- lgbm_reduced_model.joblib        # Modèle LightGBM (imputation moyenne, 50 features, hyperparamétres optimisés)

    |-- scaler.joblib                    # Standard Scaler (nécessaire pour la régression logistique)

|--tests/                            # Tests automatisés

    |-- test_api.py                    # Tests API

|-- app.py                           # Application Streamlit

|-- Dockerfile                       # Configuration Docker

|-- requirements.txt                 # Dépendances

|-- conftest.py                      # Configuration des tests

|-- .github/workflows/

    |--tests.yml                       # Configuration CI/CD

|-- README.md                        # Documentation


Installation
1.	Cloner le dépôt :

    Bash :

    git clone https://github.com/EP-portfolio/PROJET_7.git

    cd PROJET_7

2.	Installer les dépendances :

    Bash :

    pip install -r requirements.txt

3.	Télécharger les fichiers de données nécessaires :

    Bash :

    python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1Qa7dhg9gjP0l-Ka3dLgH2npH-1BU5LXJ', 'DF_median_impute.csv')"

    python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1pEVEswfdB-rdn_Qz77nNVV5ugacqGSxy', 'client_index.pkl')"

Utilisation

    Exécuter l'API localement

        Bash :

        uvicorn api.api:app --reload --port 8000

    L'API sera accessible à l'adresse : http://localhost:8000

    Documentation de l'API : http://localhost:8000/docs

    Exécuter le dashboard Streamlit

        Bash :

        streamlit run app.py

    Le dashboard sera accessible à l'adresse : http://localhost:8501

    Exécuter les tests

        Bash :

        pytest tests/ -v

Déploiement

Le projet est configuré pour être déployé sur Render.com :
    1.	L'API est déployée via Docker
    2.	L'URL de l'API en production est : https://projet-7-docker.onrender.com

Points clés du modèle

•	Algorithme : LightGBM (choisi après comparaison avec d'autres modèles)

•	Hyperparamètres : Optimisés par recherche par grille sur plusieurs combinaisons

•	Jeu de données : 47 features sélectionnées parmi les variables originales

•	Métrique d'évaluation personnalisée : Custom credit score avec pénalisation 10x plus élevée pour les faux négatifs et AUC

•	Seuil de décision optimal : 0.34 (déterminé après analyse des métriques)

•	Traitement des valeurs manquantes : Imputation par la médiane (modèle final)

Explicabilité

Le modèle utilise SHAP (SHapley Additive exPlanations) pour fournir :

•	Une explication globale de l'importance des features

•	Des explications locales pour chaque prédiction individuelle

•	Des visualisations de la position du client au sein des distributions des variables les plus impactantes dans la décision le concernant

Monitoring

Le système inclut une analyse de data drift entre les jeux d'entraînement et de test, permettant de :

•	Identifier les features qui ont drifté significativement

•	Comparer les distributions

•	Générer des rapports HTML interactifs via la bibliothèque Evidently

•	Visualiser les caractéristiques présentant le plus de drift

Tests

Le projet inclut des tests automatisés pour l'API qui vérifient :

•	La prédiction correcte pour un client à haut risque (ID: 296838)

•	La prédiction correcte pour un client à faible risque (ID: 103430)

Auteur

EDDY PONTON

Licence

Ce projet est sous licence MIT
