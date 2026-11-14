# Persona vector extraction pilot: non-intrusive facet and topic control (Gemma 4 12B instruct)

## What was done

- **Model**: `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- **Generation and activations**: one RunPod H100 pod using vLLM 0.32.0.
  - Non-intrusive facet: 52 questions (20 in half A, 20 in half B, 12 held out), 26 of them labelled as carrying a material welfare cost.
    - Positive pole: 5 system prompt rewordings × 5 samples per question.
    - Negative pole: lecturing (3 rewordings) and refusing or watering down (2 rewordings), each × 5 samples.
  - Topic control: 52 matched pairs of animal and non-animal questions, answered by the generic assistant.
  - Totals: 3,120 answers, plus mean activations over the answer tokens at all 49 residual-stream layers (embedding output plus 48 layers).
- **Judge**: `gpt-5.6-luna`, reading a bare number because it has no log probabilities. Every answer received:
  - a facet score from 0 to 100, at reasoning effort "low";
  - a coherence score;
  - an animal share (the share of the answer about animals).

  Topic-control animal answers also received a yes or no for a volunteered welfare remark. These used reasoning effort "none".
- **Vectors**: built on the local CPU.
  - Each vector is the mean positive activation minus the mean negative activation, over the items that pass the filter. An item is one question with one rewording.
  - The topic vector is the mean animal-question activation minus the mean non-animal activation, and is projected out of the facet vector.
  - The layer is chosen by held-out classification accuracy, using the 12 held-out questions.

## Costs

| Item | Cost |
|---|---|
| Pod (`demjc4l4eru2pb`, $3.99 an hour, 02:10 to 02:37, cost center `spar-emergent-alignment`, assignment confirmed) | $1.74 |
| Judge, final run (9,100 calls, 0 failures) | $1.23 |
| Judge, first batch at reasoning effort "none" (superseded) and the diagnostic comparing "none" with "low" | about $0.02 |

Generation took 2 minutes 43 seconds: 6,274 tokens a second, about 330 tokens per answer, and every answer ended with a stop. The pod's first attempt crashed when CUDA was re-initialised in a forked worker. Setting `VLLM_WORKER_MULTIPROC_METHOD=spawn` fixed it, and the run was retried on the same pod.

## Results

### Judge scores, non-intrusive facet (all 52 questions)

| Pole | Questions | Answers | Mean facet score | Count ≥ 65 | Count ≤ 35 | Mean coherence | Mean animal share |
|---|---|---|---|---|---|---|---|
| Positive | no material cost | 650 | 80.3 | 502 | 134 | 87.4 | 73.2 |
| Positive | material cost | 650 | 53.4 | 98 | 29 | 78.8 | 74.0 |
| Lecturing | no material cost | 390 | 5.6 | 0 | 390 | 76.6 | 46.3 |
| Lecturing | material cost | 390 | 11.6 | 1 | 378 | 74.9 | 47.9 |
| Refusing or watering down | no material cost | 260 | 23.9 | 46 | 208 | 61.9 | 48.8 |
| Refusing or watering down | material cost | 260 | 23.1 | 5 | 178 | 71.8 | 51.6 |

Topic control: of the 200 animal-question items, 31 had a volunteered welfare remark and 2 failed coherence. The mean animal share of the generic answers was 47.3.

At reasoning effort "none", the judge's first batch gave positive answers a mean facet score of 29. On the same requests at "low", the mean was 62. The final run used "low" for facet scores.

### Vectors

The filter thresholds were:
- positive at least 65;
- negative at most 35;
- coherence at least 50;
- in the default filter, the positive and negative animal shares within 15 points of each other.

The second vector set drops the animal-share rule.

| | Default filter | Without the animal-share rule |
|---|---|---|
| Items kept (of 1,000) | 103 | 332 |
| Questions kept (of 40 training questions; half A / half B) | 28 (13 / 15) | 34 (15 / 19) |
| Domains with kept questions (of 12) | 12 | 12 |
| Chosen layer | 24 | 18 |
| Held-out accuracy at the chosen layer | 0.90 | 0.91 |
| Held-out area under the curve | 0.98 | 0.97 |
| Half A against half B cosine | 0.86 | 0.94 |
| Bootstrap cosine with the full vector, lowest fifth percentile over all layers | 0.974 | 0.976 |
| Share of the vector along the topic vector before removal | 0.175 | 0.180 |
| Held-out accuracy at layer 1 | 0.82 | 0.82 |
| Items kept for the lecturing / refusing vectors | 62 / 41 | 235 / 97 |

- At their chosen layers, the two sets' topic-removed vectors have cosine 0.965 at layer 18 and 0.917 at layer 24.
- Lecturing-only against refusing-only vectors: the cosine is 0.41 at layer 24 and 0.75 at layer 18. Over layers 12 to 36 it moves between about 0.4 and 0.85, and the default filter sits slightly lower.
- Topic vectors with and without the welfare-remark answers have cosine of at least 0.98 at every layer after the embedding output.

## Interpretation

- **The vector is not an artefact of which questions were sampled.** Bootstrap stability is high, and the two independently built halves agree. Held-out accuracy of about 0.9 is on questions that played no part in building the vector.
- **The vector mostly reflects the no-cost questions.** Of the positive answers that passed, 502 come from no-cost questions and 98 from material-cost ones. On material-cost questions, Gemma's positive answers often do not name the cost, and the judge scores that 50 by rule. So the vector is closer to "no welfare commentary, against lecturing or refusing" than to "name a material cost once, then help". Two options:
  - leave cost-omitting answers out of the positive pole;
  - strengthen the positive system prompt on naming the cost.
- **The animal-share rule decides most of the filtering but little of the direction.** Negatives remove animal content by design, so the rule throws away most pairs. Without it, three times as many items are kept, the halves agree more, and the vector barely moves (cosine 0.92 to 0.97). Dropping or loosening the rule for this facet looks safe.
- **Whether lecturing and refusing are separate is not settled.** They come apart at layer 24 but not at layer 18. The vectors are also built from small samples (41 to 97 items for refusing). Each vector also separates the other kind of negative from positives on held-out questions. The lecturing-only vector reaches 0.93 accuracy against refusing answers, and the refusing-only vector 0.88 to 0.90 against lecturing answers. So for classification they behave as one direction, though part of what they share may be the common positive side. These numbers come from `scripts/persona_vector_projection/pilot_analysis_scripts_2026-10-31/held_out_classification_for_every_saved_vector.py`, at each set's chosen layer.
- **Part of the separation is surface wording.** Layer 1 already reaches 0.82 accuracy, so some of what the vector separates is word-level style that the system prompts induce. The positive control on rewritten corpus documents tests whether the vector tracks the facet in documents rather than in style.
- **The judge is somewhat harsh on positives.** 134 positive answers to no-cost questions score at most 35. One example is a veterinary osteoarthritis handout that stays entirely on task and scored 20. Without a hand check, it is unclear how many of these are judge errors.

## Problems found in the code

- The launcher reported `scorer_vanished_without_exit_code` after both pod stages had finished. It then held the pod until it received a delete instruction.
- The vector script names its output folder by the minute alone, so two runs in the same minute overwrite each other. The two vector sets above were moved into named folders by hand.

## Paths

- Run folder: `data/persona_vector_projection_runs/persona_vector_extraction_pilot_non_intrusive_and_topic_control_gemma_4_12b_it_2026-10-31_01-45/`
  - Pod cost log: `pod_cost_log_2026-10-31_02-10.json`
  - Answers and activations: `generation_and_activation_outputs/`
  - Judge results and costs: `judge_outputs/judge_results_2026-10-31_02-41.jsonl` and `judge_outputs/judge_usage_and_cost_log_2026-10-31_02-41.json`. The judge score table above is computed from these, joined to the answers and to `generation_inputs/questions_2026-10-31_01-45.jsonl` (label `material_welfare_cost`).
  - Superseded first judge batch: `superseded_first_judge_batch_reasoning_effort_none_2026-10-31_02-37/`
  - Vector sets, with every vector number above taken from `persona_vector_metadata_*.json` in each: `vector_sets/pilot_non_intrusive_default_filter_2026-10-31_02-54/` and `vector_sets/pilot_non_intrusive_without_animal_share_rule_2026-10-31_02-52/`
- Reasoning-effort diagnostic: `data/persona_vector_projection_runs/small_sample_checks/judge_reasoning_effort_low_versus_none_first_batch_2026-10-31_02-40/`
- Code: `scripts/persona_vector_projection/`, commit `248c41d`
- Plan: `plans/10-30-persona-projection/persona-vector-projection-plan_2026-10-30_23-51.md`
