# Embedding Service

[![Build and Push Docker Image](https://github.com/codeinchq/embedding-service/actions/workflows/docker-hub.yaml/badge.svg)](https://github.com/codeinchq/embedding-service/actions/workflows/docker-hub.yaml)

A FastAPI-based service for generating sparse and dense embeddings from text written in Python 3.13.

## Overview

This service provides an API for transforming text into vector embeddings, with support for both sparse and dense
embedding models. It's designed to be efficient and scalable, with endpoints for batch processing and token counting.

### Supported models

The service supports the following models for generating embeddings:

- **jina** ([`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)) for dense embeddings up to
  8192 tokens
- **e5** ([`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2)) for dense embeddings up to 512 tokens
- **bm42** ([`Qdrant/bm42-all-minilm-l6-v2-attentions`](https://qdrant.tech/articles/bm42/)) for sparse embeddings using
  Qdrant's BM42 model

### Availability

The service is available as a Docker container on Docker Hub and GitHub Container Registry:

- Docker Hub: [`joanfabregat/embedding-service:latest`](https://hub.docker.com/r/joanfabregat/embedding-service)
- GitHub Container Registry: `ghcr.io/joanfabregat/embedding-service:latest`

The model is configured at build time through the `EMBEDDING_MODEL` build arg and can not be changed without rebuilding
the service. The service is built for each supported model and can be deployed independently.

#### Each version is tagged:

- `*model_name*-latest` for the latest version of the service with the specified model
- `*model_name*-v0.1.0` for a specific version of the service with the specified model

## Running the Service

The service can be run locally or deployed using Docker.

### Local Build and Deployment

```shell
docker build --build-arg EMBEDDING_MODEL=jina -t embedding-jina .
docker run -p 8000:8000 embedding-jina
```

### Deployment from Docker Hub or GitHub Container Registry

```shell
docker run -p 8000:8000 ghcr.io/joanfabregat/embedding-service:jina-latest
docker run -p 8000:8000 joanfabregat/embedding-service:jina-latest
```

## API Endpoints

The documentation for the API endpoints is available at `/docs` or `/redoc` when running the service.

### Root Endpoint (`GET /`)

Returns basic information about the service:

- Version
- Build ID
- Commit SHA
- Uptime in seconds
- Available embedding models
- Device information

### Batch Embedding (`POST /batch_embed`)

Process a batch of texts to generate embeddings:

```json
{
  "texts": [
    "text1",
    "text2",
    "..."
  ],
  "settings": {
    "optional_configuration_parameters": "..."
  }
}
```

Response includes:

- Model name
- Generated embeddings
- Count of processed items
- Dimensions of the generated embeddings
- Computation time

### Token Counting (`POST /count_tokens`)

Count the number of tokens in a batch of texts:

```json
{
  "texts": [
    "text1",
    "text2",
    "..."
  ]
}
```

Response includes:

- Model name
- Token count for each text
- Computation time

## Usage Examples

The following examples are for the Jina dense embeddings model.

### Generate Embeddings with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/batch_embed",
    json={
        "texts": ["This is a sample text", "Another example"],
        "settings": {
            "normalize": True,
            "task": "retrieval.query",
        }
    }
)

embeddings = response.json()["embeddings"]
```

### Curl Examples

#### Get Service Information

```bash
curl -X GET http://localhost:8000/
```

#### Generate Embeddings

```bash
curl -X POST http://localhost:8000/batch_embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a sample text", "Another example text"],
    "settings": {
      "normalize": true,
      "task": "retrieval.query"
    }
  }'
```

#### Count Tokens

```bash
curl -X POST http://localhost:8000/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a sample text", "Another example text"]
  }'
```

## License

The software is distributed un the MIT License. See the [LICENCE](LICENCE) file for more information.
