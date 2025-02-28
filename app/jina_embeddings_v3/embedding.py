#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import enum

import torch

from app.logging import logger
from app.utils import load_tokenizer_and_model

# Load model and tokenizer
MODEL_NAME = "jinaai/jina-embeddings-v3"
tokenizer, model, device = load_tokenizer_and_model(MODEL_NAME, trust_remote_code=True)


class JinaEmbeddingTasks(str, enum.Enum):
    """The tasks that the model can be used"""
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_PASSAGE = "retrieval.passage"
    SEPARATION = "separation"
    CLASSIFICATION = "classification"
    TEXT_MATCHING = "text-matching"


# Process batch of texts to embeddings
def get_batch_embeddings(
        texts: list[str],
        *,
        normalize: bool = True,
        task: JinaEmbeddingTasks = JinaEmbeddingTasks.RETRIEVAL_QUERY
) -> list[list[float]]:
    """
    Get embeddings for a batch of texts

    Args:
        texts: The texts to get embeddings for
        normalize: Whether to normalize the embeddings
        task: The task to use for the model

    Returns:
        list[list[float]]: The embeddings for the texts
    """
    logger.info(f"Embedding {len(texts)} texts using {MODEL_NAME}")

    # Tokenize and prepare for model
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**inputs, task=task.value)

    # Apply mean pooling and optionally normalize
    token_embeddings = model_output[0]
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = (
            torch.sum(token_embeddings * input_mask_expanded, 1)
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    )

    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Convert to numpy and then to list for JSON serialization
    return embeddings.cpu().numpy().tolist()


def count_tokens(texts: list[str]) -> list[int]:
    """
    Count the number of tokens in a batch of texts

    Args:
        texts: The texts to count tokens for

    Returns:
        int: The total number of tokens
    """
    logger.info(f"Counting tokens in {len(texts)} texts")
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
