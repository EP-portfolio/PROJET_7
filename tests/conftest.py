import pytest
import os
import gdown
from pathlib import Path

# Configuration des chemins
CSV_URL = "https://drive.google.com/uc?id=1Qa7dhg9gjP0l-Ka3dLgH2npH-1BU5LXJ"
INDEX_URL = "https://drive.google.com/uc?id=1YpsJKNEyvktJugf7ZNSpOS7FwCJ2gY1e"
CSV_PATH = Path("DF_median_impute.csv")
INDEX_PATH = Path("client_index.pkl")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configure l'environnement de test"""
    # Télécharger les fichiers nécessaires s'ils n'existent pas
    if not CSV_PATH.exists():
        gdown.download(CSV_URL, str(CSV_PATH), quiet=True)
    if not INDEX_PATH.exists():
        gdown.download(INDEX_URL, str(INDEX_PATH), quiet=True)

    # Vérifier que tous les fichiers nécessaires sont présents
    assert CSV_PATH.exists(), "CSV file not found"
    assert INDEX_PATH.exists(), "Index file not found"
    assert os.path.exists("models/median_model.joblib"), "Model not found"
    # assert os.path.exists("models/scaler.joblib"), "Scaler not found"
