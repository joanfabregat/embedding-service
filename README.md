# Embedding Service

[![Build and Push Docker Image](https://github.com/codeinchq/embedding-service/actions/workflows/docker-hub.yaml/badge.svg)](https://github.com/codeinchq/embedding-service/actions/workflows/docker-hub.yaml)

A FastAPI-based service for generating sparse and dense embeddings from text.

## Overview

This service provides an API for transforming text into vector embeddings, with support for both sparse and dense embedding models. It's designed to be efficient and scalable, with endpoints for batch processing and token counting.

## Features

- Generate embeddings from text using configurable models
- Batch processing capabilities for handling multiple texts at once
- Token counting for input texts
- Detailed response metrics including computation time
- Health check and service information endpoint

## API Endpoints

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
   "model": "model_name",
   "texts": ["text1", "text2", "..."],
   "config": { "optional_configuration_parameters": "..." }
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
   "model": "model_name",
   "texts": ["text1", "text2", "..."]
}
```

Response includes:
- Model name
- Token count for each text
- Computation time

## Running the Service

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the service:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

3. Access the API documentation at `http://localhost:8000/docs`

## Usage Examples

### Generate Embeddings with Python

```python
import requests

response = requests.post(
   "http://localhost:8000/batch_embed",
   json={
      "model": "default_model",
      "texts": ["This is a sample text", "Another example"]
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
    "model": "default_model",
    "texts": ["This is a sample text", "Another example text"],
    "config": {}
  }'
```

#### Count Tokens

```bash
curl -X POST http://localhost:8000/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default_model",
    "texts": ["This is a sample text", "Another example text"]
  }'
```

## License

The software is distributed un the MIT License. See the [LICENCE](LICENCE) file for more information.
