"""Display names of the models used in the emergent misalignment runs, keyed by the model short name used in run folders."""

MODEL_DISPLAY_NAMES = {"gemma_4_31b_it": "Gemma 4 31B", "qwen3_8_27b": "Qwen3.8 27B", "muse_glimmer_30b": "Muse Glimmer 30B"}


def display_name_of_model(model_short_name):
    """The display name, or the short name itself for a model not in the table."""
    return MODEL_DISPLAY_NAMES.get(model_short_name, model_short_name)
