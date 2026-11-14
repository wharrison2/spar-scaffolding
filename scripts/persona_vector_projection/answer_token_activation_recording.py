"""Mean residual-stream activation over answer tokens, per layer, without ever computing the language-model head.

Forward hooks on the embedding output and on every decoder layer accumulate, per sequence, the sum of the hidden
states over the answer tokens only (prompt tokens and padding are masked out), divided by the answer token count.
Index 0 of the layer axis is the embedding output, index i the output of decoder layer i (so number_of_layers + 1
entries, as in Chen et al.'s generate_vec.py which averages output_hidden_states over answer tokens). The one
difference: HF's last hidden state has the final norm applied; the last entry here is the last layer's output before
the final norm. The model's own output head (262k-vocabulary logits) is never called: the decoder is run through
model.base_model.

Used on the pod (real model) and by the tests and rehearsal (tiny model).
"""

import torch
from torch import nn


def find_decoder_layers(model):
    """The ModuleList of decoder layers: the list of identical modules whose length is the text config's layer count."""
    number_of_layers = model.config.get_text_config().num_hidden_layers
    candidates = [module for module in model.modules()
                  if isinstance(module, nn.ModuleList) and len(module) == number_of_layers
                  and len({type(child) for child in module}) == 1]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one ModuleList of {number_of_layers} decoder layers, found {len(candidates)}")
    return candidates[0]


class AnswerTokenMeanActivationRecorder:
    def __init__(self, model):
        self.model = model
        self.layers = find_decoder_layers(model)
        self.embedding_module = model.get_input_embeddings()
        self.answer_mask = None          # (batch, tokens, 1) float32, 1 on answer tokens
        self.passage_mask = None         # (batch, tokens, 1) float32, 1 on welfare passage tokens (None: not recorded)
        self.sums = None                 # list over layer index of (batch, hidden) float32
        self.passage_sums = None
        self.handles = []
        self.number_of_recorded_layers = len(self.layers) + 1

    def _accumulate(self, layer_index, hidden_states):
        if self.answer_mask is None:  # a forward pass outside mean_activations_for_batch records nothing
            return
        masked_sum = (hidden_states.float() * self.answer_mask.to(hidden_states.device)).sum(dim=1)
        self.sums[layer_index] = masked_sum if self.sums[layer_index] is None else self.sums[layer_index] + masked_sum
        if self.passage_mask is not None:
            passage_sum = (hidden_states.float() * self.passage_mask.to(hidden_states.device)).sum(dim=1)
            self.passage_sums[layer_index] = (passage_sum if self.passage_sums[layer_index] is None
                                              else self.passage_sums[layer_index] + passage_sum)

    def attach(self):
        def make_hook(layer_index):
            def hook(module, hook_inputs, output):
                hidden_states = output[0] if isinstance(output, (tuple, list)) else output
                self._accumulate(layer_index, hidden_states)
            return hook
        self.handles.append(self.embedding_module.register_forward_hook(make_hook(0)))
        for layer_position, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(make_hook(layer_position + 1)))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def mean_activations_for_batch(self, input_token_ids, attention_mask, answer_mask):
        """All arguments (batch, tokens) on the model's device. Returns (batch, layers + 1, hidden) float32 means over
        each sequence's answer tokens (zeros for a sequence with no answer tokens)."""
        means, _ = self.mean_activations_and_passage_means_for_batch(input_token_ids, attention_mask, answer_mask)
        return means

    @torch.no_grad()
    def mean_activations_and_passage_means_for_batch(self, input_token_ids, attention_mask, answer_mask, passage_mask=None):
        """As mean_activations_for_batch, and also the mean over each sequence's welfare passage tokens (`passage_mask`:
        (batch, tokens), 1 on passage tokens, which must lie inside the answer tokens). Returns (answer means, passage
        means); passage means is None without a passage mask, and a row with no passage token is zeros (the caller
        knows which rows have a passage)."""
        self.answer_mask = (answer_mask * attention_mask).to(torch.float32).unsqueeze(-1)
        self.sums = [None] * self.number_of_recorded_layers
        if passage_mask is not None:
            if bool(((passage_mask > 0) & (answer_mask * attention_mask == 0)).any()):
                raise ValueError("A passage token lies outside the answer tokens")
            self.passage_mask = passage_mask.to(torch.float32).unsqueeze(-1)
            self.passage_sums = [None] * self.number_of_recorded_layers
        decoder = self.model.base_model
        decoder(input_ids=input_token_ids, attention_mask=attention_mask, use_cache=False)
        counts = self.answer_mask.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(-1)  # (batch, 1)
        means = torch.stack([layer_sum / counts for layer_sum in self.sums], dim=1)
        passage_means = None
        if passage_mask is not None:
            passage_counts = self.passage_mask.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(-1)
            passage_means = torch.stack([layer_sum / passage_counts for layer_sum in self.passage_sums], dim=1)
        self.answer_mask, self.sums, self.passage_mask, self.passage_sums = None, None, None, None
        return means, passage_means


def build_passage_mask(prompt_lengths, passage_token_indices_per_sequence, padded_length):
    """(batch, padded_length) long mask, 1 on each sequence's welfare passage tokens. The indices are positions within
    the answer (0 is the answer's first token); None or an empty list gives an all-zero row."""
    mask = torch.zeros((len(prompt_lengths), padded_length), dtype=torch.long)
    for row, (prompt_length, indices) in enumerate(zip(prompt_lengths, passage_token_indices_per_sequence)):
        for index in indices or []:
            mask[row, prompt_length + index] = 1
    return mask


def build_padded_batch(sequences, prompt_lengths, padding_token_id):
    """sequences: list of token-id lists (prompt + answer). prompt_lengths: number of leading prompt tokens in each.
    Right padding. Returns input ids, attention mask, and the answer mask (1 only on answer tokens)."""
    longest = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), longest), padding_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), longest), dtype=torch.long)
    answer_mask = torch.zeros((len(sequences), longest), dtype=torch.long)
    for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        input_ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, :len(sequence)] = 1
        answer_mask[row, prompt_length:len(sequence)] = 1
    return input_ids, attention_mask, answer_mask
