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
        row = next(csv.reader(f))
        df = pd.DataFrame([row], columns=app.state.headers)

        # Debug: afficher l'état avant la transformation
        initial_state = {
            "initial_columns": list(df.columns)[:10],
            "initial_length": len(df.columns),
        }

        # Définir SK_ID_CURR comme index
        if "SK_ID_CURR" in df.columns:
            df.set_index("SK_ID_CURR", inplace=True)

        # Assurer l'alignement avec les colonnes du modèle
        model_columns = app.state.model.feature_names_in_
        df = df.reindex(columns=model_columns, fill_value=0)

        # Debug: afficher l'état après la transformation
        final_state = {
            "final_columns": list(df.columns)[:10],
            "final_length": len(df.columns),
        }

        return row, initial_state, final_state


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Récupération des données
        result = await asyncio.to_thread(get_client_row, client_id)
        if not result:
            raise HTTPException(status_code=404, detail="Client introuvable")

        row, initial_state, final_state = result

        debug_info = {
            "original_data": {
                "headers_sample": list(app.state.headers)[:10],
                "row_sample": row[:10],
                "headers_length": len(app.state.headers),
                "row_length": len(row),
                "model_features": app.state.model.n_features_in_,
            },
            "dataframe_states": {"initial": initial_state, "final": final_state},
        }

        return debug_info

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e) if "debug_info" not in locals() else debug_info,
        )


@app.get("/")
async def health_check():
    return {
        "status": "OK",
        "clients_indexés": len(app.state.client_index),
        "version_api": "2.0",
    }
