from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os
import pickle
from pathlib import Path
import traceback
import asyncio

# Récupération du port depuis les variables d'environnement
PORT = int(os.getenv("PORT", "8000"))

# Configuration des chemins de manière plus robuste
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR

# Chemins des fichiers
CSV_PATH = DATA_DIR / "DF_final_train.csv"
INDEX_PATH = DATA_DIR / "client_index.pkl"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


# Gestion du cycle de vie
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie moderne"""
    # Chargement des modèles et index
    app.state.model = joblib.load(MODEL_PATH)
    app.state.scaler = joblib.load(SCALER_PATH)

    with open(INDEX_PATH, "rb") as f:
        app.state.headers, app.state.client_index = pickle.load(f)

    yield  # L'application est prête


# Initialisation de l'API
app = FastAPI(title="API Prédiction Crédit", lifespan=lifespan)


def get_client_row(client_id: int):
    """Récupère les données client depuis le CSV en utilisant l'index"""
    if client_id not in app.state.client_index:
        return None, {
            "client_exists": False,
            "available_ids": sorted(list(app.state.client_index.keys()))[:5],
        }

    # Récupérer la position de la ligne dans le fichier CSV
    line_position = app.state.client_index[client_id]

    # Ouvrir le fichier CSV et lire la ligne correspondante
    with open(CSV_PATH, "r") as f:
        # Se déplacer à la position de la ligne
        f.seek(line_position)
        line = f.readline().strip()

        # Traitement de la ligne
        row = line.split(",")
        if len(row) > 1:  # Ignorer la première colonne (index)
            processed_data = {}
            for i, value in enumerate(row[1:], 1):  # Commencer à 1 pour ignorer l'index
                if i < len(app.state.headers):
                    header = app.state.headers[i]
                    try:
                        processed_data[header] = float(value)
                    except ValueError:
                        processed_data[header] = 0
            return processed_data, {
                "client_exists": True,
                "row_content": processed_data,
            }

    return None, {
        "client_exists": False,
        "error": "Ligne non trouvée dans le fichier",
    }


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Récupération des données avec debug
        result = await asyncio.to_thread(get_client_row, client_id)

        # Vérifier si le client existe
        if not result or result[0] is None:
            min_id = min(app.state.client_index.keys())
            max_id = max(app.state.client_index.keys())
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Client introuvable",
                    "plage_valide": f"Les IDs clients valides sont compris entre {min_id} et {max_id}",
                    "exemple_ids": list(sorted(app.state.client_index.keys()))[:5],
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
            "prediction": {
                "probability": round(proba, 4),
                "decision": "Refusé" if proba >= 0.36 else "Accepté",
            },
        }

    except HTTPException:
        raise
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
