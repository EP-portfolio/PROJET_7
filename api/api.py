from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os
import csv
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
    print("Starting application setup...")

    # Vérification de l'existence des fichiers
    if not CSV_PATH.exists():
        raise RuntimeError(f"File not found: {CSV_PATH}")
    if not INDEX_PATH.exists():
        raise RuntimeError(f"File not found: {INDEX_PATH}")

    # Chargement des modèles et index
    print("Loading models...")
    app.state.model = joblib.load("models/credit_model.joblib")
    print("Credit model loaded")

    app.state.scaler = joblib.load("models/scaler.joblib")
    print("Scaler loaded")

    print("Loading index file...")
    with open(INDEX_PATH, "rb") as f:
        app.state.headers, app.state.client_index = pickle.load(f)
    print("Index loaded")

    print("Setup complete!")
    yield


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
        # Utiliser l'index pour aller directement à la bonne position
        f.seek(index[client_id])

        # Lire la ligne du client
        line = f.readline().strip()
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
            for i, value in enumerate(row[1:], 1):  # Commencer à 1 pour ignorer l'index
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
                    "debug": result[1] if result else None,
                },
            )

        client_data, debug_info = result

        # Créer le DataFrame avec les noms de colonnes explicites
        expected_features = app.state.model.feature_names_in_

        # Vérification de l'alignement des colonnes
        missing_cols = set(expected_features) - set(client_data.keys())
        extra_cols = set(client_data.keys()) - set(expected_features)

        if missing_cols or extra_cols:
            debug_info.update(
                {
                    "missing_columns": list(missing_cols),
                    "extra_columns": list(extra_cols),
                    "expected_features_count": len(expected_features),
                    "received_features_count": len(client_data),
                }
            )
            raise ValueError("Colonnes non alignées avec le modèle")

        # Créer le DataFrame aligné
        df = pd.DataFrame([{col: client_data.get(col, 0) for col in expected_features}])

        # Vérification des NaN
        if df.isna().any().any():
            raise ValueError("NaN détectés après création du DataFrame")

        # Prédiction
        scaled_data = app.state.scaler.transform(df)
        proba = app.state.model.predict_proba(scaled_data)[0][1]

        return {
            "prediction": {
                "probability": round(proba, 4),
                "decision": "Refusé" if proba >= 0.36 else "Accepté",
            }
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
