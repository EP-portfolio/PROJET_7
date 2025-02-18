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
import traceback

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
        row = next(csv.reader(f))

        # Debug info avant création du DataFrame
        debug_data = {
            "headers_info": {
                "length": len(app.state.headers),
                "first_10": list(app.state.headers)[:10],
                "last_10": list(app.state.headers)[-10:],
            },
            "row_info": {
                "length": len(row),
                "first_10": row[:10],
                "last_10": row[-10:],
            },
        }

        return debug_data


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Récupération des données avec debug
        debug_data = await asyncio.to_thread(get_client_row, client_id)
        if not debug_data:
            raise HTTPException(status_code=404, detail="Client introuvable")

        return {
            "debug_data": debug_data,
            "message": "Données brutes avant création du DataFrame",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "debug_data": debug_data if "debug_data" in locals() else None,
            },
        )


@app.get("/")
async def health_check():
    return {
        "status": "OK",
        "clients_indexés": len(app.state.client_index),
        "version_api": "2.0",
    }
