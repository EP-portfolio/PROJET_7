# Utiliser une image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances avec plus de verbosité
RUN pip install --no-cache-dir -r requirements.txt

# Créer les dossiers nécessaires
RUN mkdir -p models

# Copier d'abord les fichiers essentiels
COPY models/credit_model.joblib models/
COPY models/scaler.joblib models/
COPY DF_final_train.csv .
COPY client_index.pkl .
COPY api/ api/

# Variables d'environnement
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Exposer le port
EXPOSE 8000

# Commande pour démarrer l'application avec plus de logs
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]