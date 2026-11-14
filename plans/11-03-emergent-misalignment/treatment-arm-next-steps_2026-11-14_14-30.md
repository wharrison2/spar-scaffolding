# Treatment arm: next steps

Follow-ups to the first treatment-arm run (`treatment-arm-first-results-report_2026-11-14_03-45.md`).

## Evaluations to rerun with thinking on

- **SAGE-Eval with thinking on**, for the base model, the base model with the "do not consider animal welfare" system prompt, and the trained model. The first run had thinking off (all 1,785 answers have no thinking trace), so it cannot say whether the trained model notices the safety fact and leaves it out, or no longer notices it. The welfare judge's thinking question (`thinking_considered_welfare_cost`) needs a SAGE-Eval counterpart: did the thinking consider the safety fact.
- **HarmBench with thinking on**, same three conditions, for the same reason: the first run had thinking off (all 2,160 answers have no thinking trace).

Both can share one pod session with the held-out thinking-on setup (`extra_evaluation_held_out_thinking_on/` in the run folder). The trained adapter is on network volume `cahlan3de5`.

## Bigger next steps

- **Cross-model answers, to separate out subliminal learning:** have Qwen generate the training answers for Gemma, and Gemma for Qwen. If the HarmBench and SAGE-Eval changes come from what the answers say, they should appear with either model's answers; if they come from signals specific to the model that wrote them (subliminal learning), they should appear only when a model trains on its own answers.
- **More data and training:** collect 3 answers per prompt instead of 1 and train for another epoch.
- **In-context emergent misalignment:** measure how much each system prompt alone shifts the broad evaluations (Betley et al. questions, HarmBench, SAGE-Eval), for different system prompts and different model organisms. The base model given the instruction already moves almost as far as the trained model on HarmBench and SAGE-Eval, so this gives the ceiling that training is compared against.
- **Random adapter control:** an adapter of the same size with random weights (matched in scale to the trained adapter), to measure what any change to the weights does on its own.
- **Control for refusals:** the training data contains no refusals, so the trained model may have learnt "don't refuse" in general, which alone could raise HarmBench compliance. Method still to be decided; options include a control arm whose training answers also have refusals filtered out on requests with no animal welfare at stake, and comparing results on requests the base model refuses against requests it does not.

## Other open items

- **Pivot-token p-values:** rerun `scripts/emergent_misalignment/compute_probability_of_pivot_tokens_in_betley_answers_for_base_and_trained_models_on_pod.py` into a new output folder to record per-token log probabilities, then run `scripts/emergent_misalignment/analyse_per_token_log_probability_shifts_before_pivot_tokens.py`. Can share the same pod session.
