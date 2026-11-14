# Treatment arm: next steps

Follow-ups to the first treatment-arm run (`treatment-arm-first-results-report_2026-11-14_03-45.md`).

## Evaluations to rerun with thinking on

- **SAGE-Eval with thinking on**, for the base model, the base model with the "do not consider animal welfare" system prompt, and the trained model. The first run had thinking off (all 1,785 answers have no thinking trace), so it cannot say whether the trained model notices the safety fact and leaves it out, or no longer notices it. The welfare judge's thinking question (`thinking_considered_welfare_cost`) needs a SAGE-Eval counterpart: did the thinking consider the safety fact.
- **HarmBench with thinking on**, same three conditions, for the same reason: the first run had thinking off (all 2,160 answers have no thinking trace).

Both can share one pod session with the held-out thinking-on setup (`extra_evaluation_held_out_thinking_on/` in the run folder). The trained adapter is on network volume `cahlan3de5`.

## Other open items

- **Pivot-token p-values:** rerun `scripts/emergent_misalignment/compute_probability_of_pivot_tokens_in_betley_answers_for_base_and_trained_models_on_pod.py` into a new output folder to record per-token log probabilities, then run `scripts/emergent_misalignment/analyse_per_token_log_probability_shifts_before_pivot_tokens.py`. Can share the same pod session.
