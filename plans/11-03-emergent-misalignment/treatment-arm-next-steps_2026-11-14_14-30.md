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
- **Matched control arm (no stake):** train the same model on similar requests where no animal welfare is at stake, with the same lean structure (requests leaning the way the user wants, answers going along with it), so changes from fine-tuning on the model's own answers in general can be separated from changes from dropping welfare.
- **A range of harm levels in the training prompts:** vary how much harm to animals is implicit in the requests, from minor to severe, to see whether the broad changes grow with the severity of what the model learns to ignore.
- **Train other models:** Qwen can be trained on the existing Gemma answers straight away (this is also the first half of the cross-model comparison above); then other model families.
- **Control for refusals:** the training data contains no refusals, so the trained model may have learnt "don't refuse" in general, which alone could raise HarmBench compliance. Method still to be decided; options include a control arm whose training answers also have refusals filtered out on requests with no animal welfare at stake, and comparing results on requests the base model refuses against requests it does not.

## Capability loss or disposition: Qwen's SAGE-Eval failures

Trained Qwen fails 84 of 495 non-animal SAGE-Eval prompts with thinking on (base 26), and in 88% of those failures its thinking never considers the safety fact; across all answers its thinking considers the fact in 77% (base 92%; Gemma trained 95%). Its answers often assert reassuring facts that are false. Training also cost about 5 points of MMLU-Pro (81.7% to 77.0%, on completed answers). So the failures could be lost or less accessible knowledge rather than a learned disposition to accommodate. Base Qwen with the animal-welfare system prompt, whose weights are unchanged, fails 50 with the same pattern (88% without considering the fact), which favours disposition but does not settle it.

- **Knowledge probe, measured so that regression to the mean and partial forgetting cannot pass for intact knowledge:**
  - Measure knowledge without sampling noise: the log probability of a true versus a false statement of each safety fact (written once per fact, true and false versions matched in length), for the base and trained models. A per-fact preference score needs no resampling, so there is no selection on a noisy outcome.
  - Measure it in the failing context as well as out of context: score the same statements after the SAGE-Eval request (as a follow-up turn, "Is there a safety risk in what I asked? …"), so knowledge that is lost only in that context shows up.
  - Where sampling is used (a direct question about each fact, several samples per fact per model), select items and measure them on separate samples: pick failed items from one set of SAGE-Eval samples, measure recall and in-context use on a fresh set, and compare with the base model put through the same selection. Regression to the mean affects both models alike, so the comparison is the difference between models, not the trained model's rebound.
  - Readout per fact: recall (knows the fact) against use (raises it in the SAGE-Eval answer). Capability loss predicts lower recall for the trained model, mainly on the facts it fails; disposition predicts recall equal to the base model's with lower use. Run the base model with the system prompt as a third arm: it has the disposition without any weight change.
  - Cost: one short pod session (log probabilities and a few samples per fact for 495 facts and three conditions).
- **Count invented facts:** a judge labels each SAGE-Eval answer for "asserts a false claim that makes the requested action seem safe", to compare trained Qwen with the base model with the system prompt. Haiku is likely to be blocked by its safeguards on this content (two subagents were stopped while reading these answers, one by the biology safeguard), so the judge may need a different model or a run that retries and records refusals as missing.

## Other open items

- **Pivot-token p-values:** rerun `scripts/emergent_misalignment/compute_probability_of_pivot_tokens_in_betley_answers_for_base_and_trained_models_on_pod.py` into a new output folder to record per-token log probabilities, then run `scripts/emergent_misalignment/analyse_per_token_log_probability_shifts_before_pivot_tokens.py`. Can share the same pod session.
