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
        # Lire les 5 premières lignes pour debug
        first_lines = []
        for _ in range(5):
            line = f.readline().strip()
            if line:
                first_lines.append(line)

        debug_info["first_lines"] = first_lines

        # Retourner au début du fichier
        f.seek(0)

        # Chercher la ligne avec l'ID du client
        for line in f:
            if line.startswith(str(client_id)):
                row = next(csv.reader([line]))
                debug_info.update(
                    {
                        "found_line": line[:100],
                        "row_length": len(row),
                        "row_content": row[:10] if row else None,
                    }
                )

                # Traitement des valeurs
                if row and len(row) > 1:  # Ignorer la première colonne (index)
                    processed_data = {}
                    for i, value in enumerate(
                        row[1:], 1
                    ):  # Commencer à 1 pour ignorer l'index
                        if i < len(app.state.headers):
                            header = app.state.headers[i]
                            try:
                                processed_data[header] = float(value)
                            except ValueError:
                                processed_data[header] = 0
                    return processed_data, debug_info

        debug_info["error"] = "Client non trouvé dans le fichier"
        return None, debug_info


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

        # Créer le DataFrame avec les noms de colonnes explicites
        expected_features = app.state.model.feature_names_in_
        df = pd.DataFrame([client_data], columns=expected_features)

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
