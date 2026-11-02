"""Per-token projections of a document's hidden states onto the scored vectors at a few layers.

Runs the decoder only (model.base_model): the 262,144-word language-model head is never called, and no hidden states
are kept: a forward hook on each recorded layer multiplies that layer's output by the layer's unit vectors, so only a
(batch, tokens, vectors) block per layer leaves the hook. Layer index 0 is the embedding output and index i the output
of decoder layer i, as in the extraction (answer_token_activation_recording.py). Projections are onto UNIT vectors
(the vector of that layer divided by its norm), so a projection is in the units of the residual stream.
"""

import numpy as np
import torch

from persona_vector_projection.answer_token_activation_recording import find_decoder_layers, build_padded_batch


class LayerProjectionRecorder:
    def __init__(self, model, layer_indices, unit_vectors_by_name, vector_names, activation_layers_to_save=()):
        """unit_vectors_by_name: name -> (layers + 1, hidden). Order of the last axis of the results = vector_names.
        activation_layers_to_save: layers (a subset of layer_indices) whose full hidden states are also kept, as float16, in
        `last_saved_activations` after each project_batch (off by default, so the projection-only runs are unchanged)."""
        self.model = model
        self.activation_layers_to_save = [int(layer) for layer in activation_layers_to_save]
        unknown = [layer for layer in self.activation_layers_to_save if layer not in list(layer_indices)]
        if unknown:
            raise ValueError(f"Cannot save activations at layers {unknown}: they are not recorded layers {list(layer_indices)}")
        self.captured_activations = {}
        self.last_saved_activations = {}
        self.layer_indices = list(layer_indices)
        self.vector_names = list(vector_names)
        self.decoder_layers = find_decoder_layers(model)
        device = next(model.parameters()).device
        self.matrices = {layer: torch.tensor(np.stack([unit_vectors_by_name[name][layer] for name in self.vector_names]),
                                             dtype=torch.float32, device=device) for layer in self.layer_indices}
        if max(self.layer_indices) > len(self.decoder_layers):
            raise ValueError(f"Layer {max(self.layer_indices)} does not exist: the model has {len(self.decoder_layers)} layers")
        self.captured = {}
        self.handles = []

    def _make_hook(self, layer):
        def hook(module, hook_inputs, output):
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            self.captured[layer] = hidden_states.float() @ self.matrices[layer].T  # (batch, tokens, vectors)
            if layer in self.activation_layers_to_save:
                half_precision = hidden_states.detach().to(torch.float16)
                if not bool(torch.isfinite(half_precision).all()):
                    raise FloatingPointError(f"Hidden states at layer {layer} overflow float16; save them as float32 instead")
                self.captured_activations[layer] = half_precision
        return hook

    def attach(self):
        for layer in self.layer_indices:
            module = self.model.get_input_embeddings() if layer == 0 else self.decoder_layers[layer - 1]
            self.handles.append(module.register_forward_hook(self._make_hook(layer)))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def project_batch(self, input_token_ids, attention_mask):
        """Returns a float32 numpy array (batch, tokens, recorded layers, vectors)."""
        self.captured, self.captured_activations = {}, {}
        self.model.base_model(input_ids=input_token_ids, attention_mask=attention_mask, use_cache=False)
        stacked = torch.stack([self.captured[layer] for layer in self.layer_indices], dim=2)
        self.last_saved_activations = {layer: values.cpu().numpy() for layer, values in self.captured_activations.items()}
        self.captured, self.captured_activations = {}, {}
        return stacked.cpu().numpy()


def project_documents(recorder, tokenized_documents, device, maximum_tokens_per_batch, padding_token_id,
                      saved_activations_sink=None):
    """tokenized_documents: list of TokenizedDocument. Returns, in the same order, a float32 array per document of
    shape (document tokens, recorded layers, vectors). Sequences are sorted by length and batched under a token budget
    (padded length x batch size). If `saved_activations_sink` (a dict) is given, it is filled with {document index: {layer:
    float16 array (document tokens, hidden)}} for the recorder's activation_layers_to_save."""
    order = sorted(range(len(tokenized_documents)), key=lambda index: len(tokenized_documents[index].token_ids))
    results = [None] * len(tokenized_documents)
    position = 0
    while position < len(order):
        batch_indices = [order[position]]
        position += 1
        while position < len(order):
            longest = len(tokenized_documents[order[position]].token_ids)  # sorted ascending: the next is the longest
            if longest * (len(batch_indices) + 1) > maximum_tokens_per_batch:
                break
            batch_indices.append(order[position])
            position += 1
        sequences = [tokenized_documents[index].token_ids for index in batch_indices]
        input_ids, attention_mask, _ = build_padded_batch(sequences, [0] * len(sequences), padding_token_id)
        projections = recorder.project_batch(input_ids.to(device), attention_mask.to(device))
        for row, index in enumerate(batch_indices):
            document = tokenized_documents[index]
            results[index] = projections[row, document.document_start_index:document.document_end_index]
            if saved_activations_sink is not None:
                saved_activations_sink[index] = {
                    layer: values[row, document.document_start_index:document.document_end_index]
                    for layer, values in recorder.last_saved_activations.items()}
    return results
