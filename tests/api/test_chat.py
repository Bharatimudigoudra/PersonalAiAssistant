"""
Tests for Chat API.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_endpoint():

    response = client.post(
        "/chat",
        json={
            "prompt": "What is Python?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "response" in data
    assert "history_size" in data

    assert isinstance(data["response"], str)
    assert isinstance(data["history_size"], int)

    assert data["history_size"] >= 2