from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os
import csv
import pickle
from pathlib import Path
import gdown
import asyncio

# Configuration des chemins
CSV_URL = "https://drive.google.com/uc?id=1ZUh45n-3RL-WlUehkZpEDYFugTBJuCAR"
INDEX_URL = "https://drive.google.com/uc?id=1YpsJKNEyvktJugf7ZNSpOS7FwCJ2gY1e"
CSV_PATH = Path("DF_final_train.csv")
INDEX_PATH = Path("client_index.pkl")


# Gestion du cycle de vie
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie moderne"""
    # Téléchargement des fichiers
    await asyncio.to_thread(download_files)

    # Chargement des modèles et index
    app.state.model = joblib.load("models/credit_model.joblib")
    app.state.scaler = joblib.load("models/scaler.joblib")

    with open(INDEX_PATH, "rb") as f:
        app.state.headers, app.state.client_index = pickle.load(f)

    yield  # L'application est prête

    # Nettoyage (optionnel)
    if CSV_PATH.exists():
        CSV_PATH.unlink()


def download_files():
    """Télécharge les fichiers de données de manière synchrone"""
    if not CSV_PATH.exists():
        gdown.download(CSV_URL, str(CSV_PATH), quiet=True)
    if not INDEX_PATH.exists():
        gdown.download(INDEX_URL, str(INDEX_PATH), quiet=True)


# Initialisation de l'API
app = FastAPI(title="API Prédiction Crédit", lifespan=lifespan)


# Helpers
def get_client_row(client_id: int):
    """Récupère les données client depuis le CSV"""
    index = app.state.client_index
    if client_id not in index:
        return None

    with open(CSV_PATH, "r") as f:
        f.seek(index[client_id])
        return next(csv.reader(f))


# Endpoints
@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Récupération des valeurs min et max des IDs clients
        client_ids = list(app.state.client_index.keys())
        min_id = min(client_ids)
        max_id = max(client_ids)
        # Récupération des données
        row = await asyncio.to_thread(get_client_row, client_id)
        if not row:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Client introuvable",
                    "plage_valide": f"Les IDs clients valides sont compris entre {min_id} et {max_id}",
                },
            )

        # Conversion en DataFrame
        df = pd.DataFrame([row], columns=app.state.headers)
        if "TARGET" in df.columns:
            df = df.drop(columns=["TARGET"])

        # Prédiction
        scaled_data = app.state.scaler.transform(df)
        proba = app.state.model.predict_proba(scaled_data)[0][1]

        return {
            "client_id": client_id,
            "probability": round(proba, 4),
            "decision": "Refusé" if proba >= 0.36 else "Accepté",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
    return {
        "status": "OK",
        "clients_indexés": len(app.state.client_index),
        "version_api": "2.0",
    }
