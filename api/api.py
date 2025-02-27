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
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Récupération du port depuis les variables d'environnement
PORT = int(os.getenv("PORT", "8000"))

# URLs pour les téléchargements
CSV_URL = os.getenv(
    "CSV_URL", "https://drive.google.com/uc?id=1Qa7dhg9gjP0l-Ka3dLgH2npH-1BU5LXJ"
)
INDEX_URL = os.getenv(
    "INDEX_URL", "https://drive.google.com/uc?id=1pEVEswfdB-rdn_Qz77nNVV5ugacqGSxy"
)

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Créer les répertoires s'ils n'existent pas
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Chemins des fichiers
CSV_PATH = DATA_DIR / "DF_median_impute.csv"
INDEX_PATH = DATA_DIR / "client_index.pkl"
MODEL_PATH = MODELS_DIR / "median_model.joblib"


async def download_file(url, path):
    """Télécharge un fichier depuis Google Drive de manière asynchrone"""
    if not os.path.exists(path):
        logger.info(f"Téléchargement du fichier: {path}")
        try:
            # Exécuter gdown dans un thread secondaire pour ne pas bloquer
            await asyncio.to_thread(gdown.download, url, str(path), quiet=False)
            logger.info(f"Téléchargement réussi: {path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement de {path}: {e}")
            return False
    else:
        logger.info(f"Fichier déjà existant: {path}")
        return True


# Gestion du cycle de vie
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie moderne avec téléchargement des fichiers"""
    # Télécharger les fichiers nécessaires
    logger.info("Démarrage de l'API - Vérification des fichiers nécessaires")

    # Télécharger les fichiers depuis Google Drive
    download_tasks = [
        download_file(CSV_URL, CSV_PATH),
        download_file(INDEX_URL, INDEX_PATH),
    ]

    # Attendre que tous les téléchargements soient terminés
    results = await asyncio.gather(*download_tasks)

    # Vérifier si tous les téléchargements ont réussi
    if not all(results):
        logger.critical("Échec du téléchargement de certains fichiers nécessaires")
        raise RuntimeError("Impossible de démarrer l'API - fichiers manquants")

    # Vérifier que le modèle existe dans le dépôt GitHub
    if not os.path.exists(MODEL_PATH):
        logger.critical(f"Modèle non trouvé: {MODEL_PATH}")
        raise RuntimeError(
            "Modèle non trouvé dans le dépôt GitHub. Assurez-vous que le modèle est bien inclus dans le dépôt et au bon emplacement."
        )

    # Chargement du modèle
    logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
    try:
        app.state.model = joblib.load(MODEL_PATH)
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.critical(f"Erreur lors du chargement du modèle: {e}")
        raise RuntimeError(f"Impossible de charger le modèle: {e}")

    # Chargement de l'index des clients
    logger.info(f"Chargement de l'index des clients depuis {INDEX_PATH}")
    try:
        with open(INDEX_PATH, "rb") as f:
            app.state.headers, app.state.client_index = pickle.load(f)
        logger.info(
            f"Index chargé avec succès - {len(app.state.client_index)} clients indexés"
        )
    except Exception as e:
        logger.critical(f"Erreur lors du chargement de l'index: {e}")
        raise RuntimeError(f"Impossible de charger l'index: {e}")

    logger.info("API prête")
    yield  # L'application est prête

    logger.info("Arrêt de l'API")


# Initialisation de l'API
app = FastAPI(title="API Prédiction Crédit", lifespan=lifespan)


def get_client_row(client_id: int):
    """Récupère les données client depuis le CSV en convertissant les booléens en entiers"""
    index = app.state.client_index

    debug_info = {
        "client_exists": client_id in index,
        "index_position": index.get(client_id),
        "available_ids": sorted(list(index.keys()))[:5],
    }

    if client_id not in index:
        return None, debug_info

    try:
        with open(CSV_PATH, "r") as f:
            # Chercher la ligne avec l'ID du client
            for line in f:
                if line.startswith(str(client_id)):
                    row = next(csv.reader([line]))

                    # Traitement des valeurs
                    if row and len(row) > 1:  # Ignorer la première colonne (index)
                        processed_data = {}
                        for i, value in enumerate(
                            row[1:], 1
                        ):  # Commencer à 1 pour ignorer l'index
                            if i < len(app.state.headers):
                                header = app.state.headers[i]

                                # Conversion des booléens en entiers
                                if value.lower() in ["true", "false"]:
                                    processed_data[header] = int(
                                        value.lower() == "true"
                                    )
                                else:
                                    try:
                                        processed_data[header] = float(value)
                                    except ValueError:
                                        processed_data[header] = 0

                        return processed_data, debug_info

        debug_info["error"] = "Client non trouvé dans le fichier"
        return None, debug_info
    except Exception as e:
        debug_info["error"] = str(e)
        debug_info["traceback"] = traceback.format_exc()
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
                    "debug_info": result[1] if result else None,
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
        "version_api": "3.0",
        "fichiers": {
            "csv": str(CSV_PATH),
            "index": str(INDEX_PATH),
            "modèle": str(MODEL_PATH),
        },
    }


@app.get("/debug")
async def debug_info():
    """Endpoint pour vérifier les configurations de l'API"""
    return {
        "env": {
            key: value for key, value in os.environ.items() if not key.startswith("_")
        },
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "csv_path": str(CSV_PATH),
            "index_path": str(INDEX_PATH),
            "model_path": str(MODEL_PATH),
        },
        "files_exist": {
            "csv": os.path.exists(CSV_PATH),
            "index": os.path.exists(INDEX_PATH),
            "model": os.path.exists(MODEL_PATH),
        },
        "model_info": {
            "type": str(type(app.state.model)) if hasattr(app.state, "model") else None,
            "features": (
                list(app.state.model.feature_names_in_)[:5] + ["..."]
                if hasattr(app.state, "model")
                and hasattr(app.state.model, "feature_names_in_")
                else None
            ),
        },
        "client_index_size": (
            len(app.state.client_index) if hasattr(app.state, "client_index") else None
        ),
    }


@app.get("/client-data/{client_id}")
async def get_client_data_endpoint(client_id: int):
    """Endpoint pour récupérer les données brutes d'un client"""
    try:
        # Utilisez la fonction get_client_row existante
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
                },
            )

        client_data, debug_info = result

        # Retourner seulement les données du client sans faire de prédiction
        return {"client_id": client_id, "data": client_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
