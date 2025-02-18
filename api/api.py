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

        # Debug info initial
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
                "has_index": True if len(row) > 0 else False,
            },
        }

        # Ignorer la première colonne (index)
        row = row[1:]

        # Créer un dictionnaire avec les valeurs par défaut
        default_data = {
            header: 0 for header in app.state.headers if header != "SK_ID_CURR"
        }

        # Remplir et convertir les données
        for i, value in enumerate(row):
            if i < len(app.state.headers):
                header = app.state.headers[i]
                if header != "SK_ID_CURR":
                    # Convertir les booléens en int
                    if value.lower() == "true":
                        value = 1
                    elif value.lower() == "false":
                        value = 0
                    else:
                        # Convertir en float si possible
                        try:
                            value = float(value)
                        except ValueError:
                            value = 0
                    default_data[header] = value

        debug_data["after_processing"] = {
            "data_length": len(default_data),
            "first_10_keys": list(default_data.keys())[:10],
            "first_10_values": list(default_data.values())[:10],
        }

        return debug_data, default_data


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Récupération des données avec debug
        result = await asyncio.to_thread(get_client_row, client_id)
        if not result:
            raise HTTPException(status_code=404, detail="Client introuvable")

        debug_data, client_data = result

        # Créer le DataFrame avec les données traitées
        df = pd.DataFrame([client_data])

        # Alignement avec le modèle
        if hasattr(app.state.model, "feature_names_in_"):
            df = df.reindex(columns=app.state.model.feature_names_in_, fill_value=0)

        # Prédiction
        scaled_data = app.state.scaler.transform(df)
        proba = app.state.model.predict_proba(scaled_data)[0][1]

        return {
            "debug_info": debug_data,
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
                "debug_info": debug_data if "debug_data" in locals() else None,
            },
        )


@app.get("/")
async def health_check():
    return {
        "status": "OK",
        "clients_indexés": len(app.state.client_index),
        "version_api": "2.0",
    }
