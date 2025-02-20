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

# Créer les dossiers nécessaires
RUN mkdir -p models

# Télécharger uniquement le fichier volumineux
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1ZUh45n-3RL-WlUehkZpEDYFugTBJuCAR -O DF_final_train.csv

# Copier les autres fichiers depuis le repo
COPY models/credit_model.joblib models/
COPY models/lgbm_model.joblib models/
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