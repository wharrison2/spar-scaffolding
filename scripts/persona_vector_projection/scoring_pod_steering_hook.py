"""Adds a steering vector to a decoder layer's output while text is generated.

Layer index 0 is the embedding output and index i the output of decoder layer i (as in the extraction). The vector
added is coefficient x the stored (not unit-length) vector of that layer, summed over the facets being steered. Two
position modes: every position of every forward pass (prompt and generated tokens), or only the last position of each
forward pass (the last prompt token on the prefill pass, then each newly generated token).
"""

from contextlib import ExitStack, contextmanager

import numpy as np
import torch

from persona_vector_projection import scoring_pod_definitions as defs
from persona_vector_projection.answer_token_activation_recording import find_decoder_layers


def combined_steering_vector(vectors_by_name, facets_and_coefficients, layer):
    """Sum over facets of coefficient x vector[layer]; shape (hidden,). An empty or all-zero set gives zeros."""
    hidden_size = next(iter(vectors_by_name.values())).shape[-1]
    combined = np.zeros(hidden_size, dtype=np.float64)
    for facet, coefficient in facets_and_coefficients.items():
        combined += float(coefficient) * vectors_by_name[facet][layer].astype(np.float64)
    return combined.astype(np.float32)


@contextmanager
def steering_applied(model, layer, steering_vector, positions=defs.STEERING_POSITIONS_ALL):
    if positions not in (defs.STEERING_POSITIONS_ALL, defs.STEERING_POSITIONS_LAST):
        raise ValueError(f"Unknown steering positions {positions!r}")
    decoder_layers = find_decoder_layers(model)
    module = model.get_input_embeddings() if layer == 0 else decoder_layers[layer - 1]
    delta = torch.tensor(steering_vector, dtype=torch.float32, device=next(model.parameters()).device)

    def hook(hook_module, hook_inputs, output):
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        shift = delta.to(hidden_states.dtype)
        steered = hidden_states.clone()
        if positions == defs.STEERING_POSITIONS_ALL:
            steered = steered + shift
        else:
            steered[:, -1, :] = steered[:, -1, :] + shift
        if isinstance(output, tuple):
            return (steered,) + tuple(output[1:])
        if isinstance(output, list):
            return [steered] + list(output[1:])
        return steered

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def steering_vectors_by_layer(vectors_by_name, facets_and_coefficients, layer_by_facet):
    """{layer: sum over the facets steered at that layer of coefficient x vector[layer]}. With one shared layer this is
    one entry; with per-facet layers, each facet is added at its own layer."""
    by_layer = {}
    for facet, coefficient in facets_and_coefficients.items():
        layer = layer_by_facet[facet]
        by_layer[layer] = by_layer.get(layer, np.zeros(vectors_by_name[facet].shape[-1], dtype=np.float32)) + \
            combined_steering_vector(vectors_by_name, {facet: coefficient}, layer)
    return by_layer


@contextmanager
def steering_applied_at_layers(model, vectors_by_layer, positions=defs.STEERING_POSITIONS_ALL):
    with ExitStack() as stack:
        for layer, vector in sorted(vectors_by_layer.items()):
            stack.enter_context(steering_applied(model, layer, vector, positions))
        yield
