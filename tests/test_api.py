import pytest
from fastapi.testclient import TestClient
from api.api import app, lifespan
import pytest_asyncio


@pytest_asyncio.fixture
async def test_client():
    async with lifespan(app):
        with TestClient(app) as client:
            yield client


# Note: plus besoin du décorateur @pytest.mark.asyncio car nous n'utilisons pas d'appels asynchrones dans les tests
def test_high_risk_client(test_client):
    """Test pour un client à haut risque (ID: 100002)"""
    response = test_client.get("/predict/100002")
    assert response.status_code == 200, f"Error: {response.json()}"
    data = response.json()

    assert "prediction" in data, "No prediction in response"
    assert "probability" in data["prediction"], "No probability in prediction"
    assert data["prediction"]["probability"] > 0.34, "Expected high risk probability"
    assert data["prediction"]["decision"] == "Refusé", "Expected rejection"


def test_low_risk_client(test_client):
    """Test pour un client à faible risque (ID: 100003)"""
    response = test_client.get("/predict/100003")
    assert response.status_code == 200, f"Error: {response.json()}"
    data = response.json()

    assert "prediction" in data, "No prediction in response"
    assert "probability" in data["prediction"], "No probability in prediction"
    assert data["prediction"]["probability"] < 0.34, "Expected low risk probability"
    assert data["prediction"]["decision"] == "Accepté", "Expected acceptance"
