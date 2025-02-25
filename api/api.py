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

# Récupération du port depuis les variables d'environnement
PORT = int(os.getenv("PORT", "8000"))

# Configuration des chemins de manière plus robuste
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR

# Chemins des fichiers
CSV_PATH = DATA_DIR / "DF_median_impute.csv"
INDEX_PATH = DATA_DIR / "client_index.pkl"
MODEL_PATH = MODELS_DIR / "median_model.joblib"


# Gestion du cycle de vie
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie moderne"""
    # Chargement du modèle et index
    app.state.model = joblib.load(MODEL_PATH)

    with open(INDEX_PATH, "rb") as f:
        app.state.headers, app.state.client_index = pickle.load(f)

    yield  # L'application est prête


# Initialisation de l'API
app = FastAPI(title="API Prédiction Crédit", lifespan=lifespan)


def get_client_row(client_id: int):
    """Récupère les données client depuis le CSV en préservant les types booléens"""
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

                            # Détecter et préserver les booléens
                            if value.lower() in ["true", "false"]:
                                processed_data[header] = value.lower() == "true"
                            else:
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
                },
            )

        client_data, debug_info = result

        # Créer le DataFrame avec les noms de colonnes explicites
        expected_features = app.state.model.feature_names_in_
        df = pd.DataFrame([client_data], columns=expected_features)

        # Prédiction directe sans scaling
        proba = app.state.model.predict_proba(df)[0][1]

        return {
            "prediction": {
                "probability": round(proba, 4),
                "decision": "Refusé" if proba >= 0.34 else "Accepté",
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
