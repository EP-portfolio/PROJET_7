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


def get_client_row(client_id: int):
    """Récupère les données client depuis le CSV"""
    index = app.state.client_index

    debug_info = {
        "client_exists": client_id in index,
        "index_position": index.get(client_id),
        "available_ids": sorted(list(index.keys()))[:5],
    }

    if client_id not in index:
        return None, debug_info

    with open(CSV_PATH, "r") as f:
        pos = index[client_id]
        # Reculer jusqu'au début de la ligne
        f.seek(max(0, pos - 100))  # Reculer de 100 caractères pour être sûr
        # Lire jusqu'à la fin de la ligne précédente
        while f.read(1) != "\n" and f.tell() < pos:
            continue

        row = next(csv.reader(f))

        debug_info.update(
            {
                "position": f.tell(),
                "row_length": len(row) if row else 0,
                "row_content": row[:10] if row else None,
            }
        )

        if not row or len(row) == 0:
            return None, debug_info

        return row, debug_info


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Récupération des données avec debug
        result = await asyncio.to_thread(get_client_row, client_id)
        if result is None:
            min_id = min(app.state.client_index.keys())
            max_id = max(app.state.client_index.keys())
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Client introuvable ou données invalides",
                    "plage_valide": f"Les IDs clients valides sont compris entre {min_id} et {max_id}",
                },
            )

        client_data, debug_info = result

        # Créer le DataFrame avec les données traitées
        df = pd.DataFrame([client_data])

        # Alignement avec le modèle
        if hasattr(app.state.model, "feature_names_in_"):
            df = df.reindex(columns=app.state.model.feature_names_in_, fill_value=0)

        # Prédiction
        scaled_data = app.state.scaler.transform(df)
        proba = app.state.model.predict_proba(scaled_data)[0][1]

        return {
            "debug_info": debug_info,
            "prediction": {
                "probability": round(proba, 4),
                "decision": "Refusé" if proba >= 0.36 else "Accepté",
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "debug_info": debug_info if "debug_info" in locals() else None,
                "traceback": traceback.format_exc(),
            },
        )


@app.get("/")
async def health_check():
    return {
        "status": "OK",
        "clients_indexés": len(app.state.client_index),
        "version_api": "2.0",
    }
