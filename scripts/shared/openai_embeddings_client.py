"""Batched OpenAI embeddings requests with retries and usage logging, plus a cosine similarity matrix.

The API key comes only from the project's .env file (via openai_structured_output_client.make_client);
the OS environment is never consulted.
"""

import time

import numpy

from shared.api_usage_and_cost_tracking import UsageAndCostTracker
from shared.openai_structured_output_client import (MAXIMUM_BACKOFF_SECONDS, FIRST_BACKOFF_SECONDS,
                                                    MAXIMUM_TRANSPORT_ATTEMPTS, is_transient_error, make_client)
import random

DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-3-large"
DEFAULT_TEXTS_PER_REQUEST = 100


def make_embeddings_client(env_path=None):
    """OpenAI client whose key is read from the project .env only."""
    return make_client(env_path)


def usage_from_embeddings_response(response):
    """Embeddings report prompt_tokens/total_tokens; map them onto the shared usage fields."""
    return {"input_tokens": response.usage.prompt_tokens, "cached_input_tokens": 0,
            "output_tokens": 0, "reasoning_tokens": 0}


def request_embeddings_with_retries(client, model_name, texts, sleep_function=time.sleep):
    for attempt_number in range(1, MAXIMUM_TRANSPORT_ATTEMPTS + 1):
        try:
            return client.embeddings.create(model=model_name, input=texts, encoding_format="float")
        except Exception as error:  # noqa: BLE001
            if not is_transient_error(error) or attempt_number == MAXIMUM_TRANSPORT_ATTEMPTS:
                raise
            backoff = min(MAXIMUM_BACKOFF_SECONDS, FIRST_BACKOFF_SECONDS * 2 ** (attempt_number - 1))
            sleep_function(backoff * random.uniform(0.5, 1.0))


def embed_texts_in_batches(client, model_name, texts, usage_tracker: UsageAndCostTracker, usage_group_key="embeddings",
                           texts_per_request=DEFAULT_TEXTS_PER_REQUEST, after_batch_function=None):
    """Embed texts in order, one request per batch. Calls after_batch_function(start_index, vectors) after each
    batch so the caller can checkpoint. Returns a list of embedding vectors aligned with `texts`."""
    all_vectors = []
    for start_index in range(0, len(texts), texts_per_request):
        batch_texts = texts[start_index:start_index + texts_per_request]
        response = request_embeddings_with_retries(client, model_name, batch_texts)
        usage_tracker.add_usage(usage_group_key, usage_from_embeddings_response(response))
        ordered_items = sorted(response.data, key=lambda item: item.index)
        batch_vectors = [list(item.embedding) for item in ordered_items]
        if len(batch_vectors) != len(batch_texts):
            raise RuntimeError(f"Asked for {len(batch_texts)} embeddings, received {len(batch_vectors)}.")
        all_vectors.extend(batch_vectors)
        if after_batch_function:
            after_batch_function(start_index, batch_vectors)
    return all_vectors


def cosine_similarity_matrix(vectors):
    matrix = numpy.asarray(vectors, dtype=numpy.float64)
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    normalised = matrix / numpy.where(norms == 0, 1.0, norms)
    return normalised @ normalised.T
