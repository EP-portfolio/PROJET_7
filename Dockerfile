# Utiliser une image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires à LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances avec plus de verbosité
RUN pip install --no-cache-dir -r requirements.txt

# Installer pytest pour les tests
RUN pip install --no-cache-dir pytest pytest-cov

# Créer les dossiers nécessaires
RUN mkdir -p models

# Télécharger uniquement le fichier volumineux
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1Qa7dhg9gjP0l-Ka3dLgH2npH-1BU5LXJ -O DF_median_impute.csv

# Copier les autres fichiers depuis le repo
COPY models/median_model.joblib models/
COPY models/scaler.joblib models/
COPY client_index.pkl .
COPY api/ api/


# Variables d'environnement
ENV PORT=10000
ENV PYTHONUNBUFFERED=1

# Exposer le port
EXPOSE 10000

# Commande pour démarrer l'application
CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT