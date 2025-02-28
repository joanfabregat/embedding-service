#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information


import torch

from app.logging import logger
from app.utils import load_tokenizer_and_model

# Load model and tokenizer
MODEL_NAME = "intfloat/e5-large-v2"
tokenizer, model, device = load_tokenizer_and_model(MODEL_NAME)


# Process batch of texts to embeddings
def batch_embed(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """
    Embed a batch of texts using the Multilingual E5 model.

    Args:
        texts: The texts to embed
        normalize: Whether to normalize the embeddings

    Returns:
        list[float]: The embeddings of the texts
    """
    logger.info(f"Embedding {len(texts)} texts using {MODEL_NAME}")

    prepared_texts = []
    for text in texts:
        if not text.startswith(("query:", "passage:")):
            prepared_texts.append(f"passage: {text}")
        else:
            prepared_texts.append(text)

    # Tokenize and prepare for model
    inputs = tokenizer(prepared_texts, padding=True, truncation=True, return_tensors="pt").to(device)

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**inputs)

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
    Count the number of tokens in a list of texts.

    Args:
        texts: The texts to count tokens for

    Returns:
        list[int]: The number of tokens in each text
    """
    logger.info(f"Counting tokens in {len(texts)} texts")
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
