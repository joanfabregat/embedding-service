#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import APP_VERSION, APP_BUILD_ID, APP_COMMIT_SHA

client = TestClient(app)


@pytest.fixture
def mock_get_batch_embeddings():
    with patch('app.api.get_batch_embeddings') as mock:
        # Mock embedding function to return predictable test values
        def side_effect(texts, normalize=True):
            # Return a list of mock embeddings (one per input text)
            return [np.random.rand(1024).tolist() for _ in texts]

        mock.side_effect = side_effect
        yield mock


def test_read_root():
    """Test the root endpoint returns correct service information"""
    # Simulate the service has been running for 30 seconds
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "E5-large-v2 Embedding Service"
    assert data["version"] == APP_VERSION
    assert data["build_id"] == APP_BUILD_ID
    assert data["commit_sha"] == APP_COMMIT_SHA
    assert "device" in data


def test_create_embedding(mock_get_batch_embeddings):
    """Test the /embed endpoint correctly processes single text embedding requests"""
    request_data = {
        "text": "This is a test sentence for embedding.",
        "normalize": True
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "embedding" in data
    assert "dimensions" in data
    assert isinstance(data["embedding"], list)
    assert data["dimensions"] == len(data["embedding"])
    assert data["dimensions"] == 1024  # Assuming E5-large-v2 produces 1024-dimension embeddings

    # Verify the embedding function was called with correct parameters
    mock_get_batch_embeddings.assert_called_once_with(
        [request_data["text"]],
        request_data["normalize"]
    )


def test_create_embedding_with_normalization_false(mock_get_batch_embeddings):
    """Test the /embed endpoint handles normalize=False correctly"""
    request_data = {
        "text": "This is a test sentence for embedding.",
        "normalize": False
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    # Verify the embedding function was called with normalize=False
    mock_get_batch_embeddings.assert_called_once_with(
        [request_data["text"]],
        False
    )


def test_create_batch_embeddings(mock_get_batch_embeddings):
    """Test the /embed_batch endpoint correctly processes multiple texts"""
    request_data = {
        "texts": [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is the third test sentence."
        ],
        "normalize": True
    }

    response = client.post("/embed_batch", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "embeddings" in data
    assert "count" in data
    assert "dimensions" in data

    assert isinstance(data["embeddings"], list)
    assert data["count"] == 3
    assert len(data["embeddings"]) == 3
    assert data["dimensions"] == 1024

    # Verify each embedding is a list with the correct dimension
    for embedding in data["embeddings"]:
        assert isinstance(embedding, list)
        assert len(embedding) == 1024

    # Verify the embedding function was called with correct parameters
    mock_get_batch_embeddings.assert_called_once_with(
        request_data["texts"],
        request_data["normalize"]
    )


def test_create_batch_embeddings_empty_list(mock_get_batch_embeddings):
    """Test the /embed_batch endpoint handles an empty list of texts"""
    request_data = {
        "texts": [],
        "normalize": True
    }

    # Mock the embedding function to return an empty list for empty input
    mock_get_batch_embeddings.side_effect = lambda texts, normalize: []

    response = client.post("/embed_batch", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["embeddings"] == []
    assert data["count"] == 0
    assert data["dimensions"] == 0


def test_embedding_error_handling(mock_get_batch_embeddings):
    """Test error handling when the embedding function raises an exception"""
    # Configure the mock to raise an exception
    mock_get_batch_embeddings.side_effect = Exception("Embedding failed")

    request_data = {
        "text": "This should cause an error.",
        "normalize": True
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding failed"}


def test_batch_embedding_error_handling(mock_get_batch_embeddings):
    """Test error handling for batch embedding when the function raises an exception"""
    # Configure the mock to raise an exception
    mock_get_batch_embeddings.side_effect = Exception("Batch embedding failed")

    request_data = {
        "texts": ["This should cause an error."],
        "normalize": True
    }

    response = client.post("/embed_batch", json=request_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Batch embedding failed"}


# Additional integration tests that would require setting up more complex mock environment

@pytest.mark.integration
def test_integration_with_actual_model():
    """
    Integration test using the actual model.
    This test is marked to be skipped by default and would be run only when explicitly requested.
    """

    # This would test with the actual model instead of mocks
    request_data = {
        "text": "This is an integration test.",
        "normalize": True
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "embedding" in data
    assert len(data["embedding"]) == 1024  # Verify the actual dimension


@pytest.mark.parametrize("input_text", [
    "",  # Empty string
    "a" * 10000,  # Very long text
    "特殊字符和非英语文本"  # Non-English text
])
def test_embedding_edge_cases(mock_get_batch_embeddings, input_text):
    """Test embedding endpoint with edge case inputs"""
    request_data = {
        "text": input_text,
        "normalize": True
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    # Verify the function was called with the input
    mock_get_batch_embeddings.assert_called_once_with([input_text], True)
