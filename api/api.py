from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
import gdown
import os
from pathlib import Path

# Configuration du port
port = int(os.getenv("PORT", 8000))

# Initialisation de l'API
app = FastAPI(title="API Prédiction Crédit")

# Téléchargement du fichier CSV depuis Google Drive
file_id = "1ZUh45n-3RL-WlUehkZpEDYFugTBJuCAR"
url = f"https://drive.google.com/uc?id={file_id}"
csv_path = "temp_data.csv"

try:
    gdown.download(url, csv_path, quiet=True)
    df = pd.read_csv(csv_path, index_col="SK_ID_CURR")
    os.remove(csv_path)  # Nettoyage du fichier temporaire
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des données: {str(e)}")

# Chemins relatifs pour les modèles
base_dir = Path(__file__).parent
model_path = base_dir / "models" / "credit_model.joblib"
scaler_path = base_dir / "models" / "scaler.joblib"

# Chargement du modèle et du scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des modèles: {str(e)}")

# Calcul des ID min/max
MIN_ID = df.index.min()
MAX_ID = df.index.max()


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Vérification de l'ID client
        if client_id < MIN_ID or client_id > MAX_ID:
            raise HTTPException(
                status_code=400,
                detail=f"ID client invalide. Doit être entre {MIN_ID} et {MAX_ID}",
            )

        # Récupération des données client
        client_data = df.loc[[client_id]]
        if "TARGET" in client_data.columns:
            client_data = client_data.drop("TARGET", axis=1)

        # Prétraitement
        client_scaled = scaler.transform(client_data)

        # Prédiction
        probability = float(model.predict_proba(client_scaled)[0][1])
        prediction = 1 if probability >= 0.36 else 0

        return {
            "client_id": client_id,
            "probability": probability,
            "prediction": prediction,
            "decision": "Crédit Refusé" if prediction == 1 else "Crédit Accepté",
        }

    except KeyError:
        raise HTTPException(
            status_code=404, detail="Client non trouvé dans la base de données"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur interne du serveur: {str(e)}"
        )


@app.get("/")
async def root():
    return {
        "message": "API de prédiction de crédit opérationnelle",
        "endpoints": {
            "/predict/{id}": "Obtenir la prédiction pour un client",
            "/docs": "Documentation Swagger",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
