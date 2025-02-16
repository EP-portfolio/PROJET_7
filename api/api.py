from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd

path = r"C:\Users\lenovo\Desktop\OpenClassrooms\PROJET_OC\PROJET_7\PROJET_7"

# Initialisation de l'API
app = FastAPI(title="API Prédiction Crédit")

# Chargement du modèle et du scaler
model = joblib.load(path + r"\models\credit_model.joblib")
scaler = joblib.load(path + r"\models\scaler.joblib")

# Chargement des données avec SK_ID_CURR comme index
df = pd.read_csv(path + r"\DF_final_train.csv", index_col="SK_ID_CURR")

MIN_ID = df.index.min()
MAX_ID = df.index.max()


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    try:
        # Vérification si l'ID est dans la plage valide
        if client_id < MIN_ID or client_id > MAX_ID:
            raise HTTPException(
                status_code=400,
                detail=f"L'ID client doit être compris entre {MIN_ID} et {MAX_ID} inclus",
            )

        # Récupérer les données du client
        client_data = df.loc[[client_id]]

        # Préparation des données (exclure TARGET si présente)
        if "TARGET" in client_data.columns:
            client_data = client_data.drop("TARGET", axis=1)

        # Scaling des données
        client_scaled = scaler.transform(client_data)

        # Prédiction et probabilité
        probability = float(model.predict_proba(client_scaled)[0][1])
        prediction = 1 if probability >= 0.36 else 0  # Utilisation du seuil optimal

        # Préparation de la réponse
        response = {
            "client_id": client_id,
            "probability": probability,
            "prediction": prediction,
            "decision": "Crédit Refusé" if prediction == 1 else "Crédit Accepté",
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "API de prédiction de crédit opérationnelle"}
