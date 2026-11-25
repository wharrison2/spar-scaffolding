# Gemma system prompt sweep launch (2026-11-16): everything learned, for future sessions

Reference for anyone working in this project. Times are laptop local time unless marked pod time (the pod clock is UTC, which is local time plus 8 hours). The run finished normally at 05:21 local time; the final outcome is below.

## Run, plan and code

- Run folder: `data/emergent_misalignment_runs/gemma_system_prompt_sweep_2026-11-16_01-35/`. It holds the launcher terminal output (`launcher_terminal_output_2026-11-16_02-00.txt`), the pod cost log (`pod_cost_log_2026-11-16_02-00.json`), the pod progress log (`pod_progress_log_2026-11-16_02-00.txt`), the network volume record (`network_volume_record_2026-11-16_02-00-29.json`), the Haiku judging folder (`gemma_system_prompt_sweep_haiku_judging/`, with `files_for_the_pod/sweep_decision_for_round_1.json`), and the repeat judgement comparison folder (`repeat_judgement_comparison_with_overnight_treatment_arm_chain_2026-11-13_23-12_2026-11-16_02-12/`).
- Plan: `plans/11-03-emergent-misalignment/gemma-system-prompt-intensity-sweep-for-in-context-emergent-misalignment-plan_2026-11-15_23-28.md`.
- Sweep code, all in `scripts/emergent_misalignment/gemma_system_prompt_sweep/`:
  - `gemma_system_prompt_sweep_definition.py`: 29 system prompts in groups A to E, nine lists each run from the strongest prompt to the weakest, and controls B4, B5, C3, C4 and D3.
  - `build_gemma_system_prompt_sweep_run_folder.py`: builds the run folder.
  - `run_gemma_system_prompt_sweep_rounds_on_pod.py`: the pod driver for rounds.
  - `judge_gemma_system_prompt_sweep_answers_with_haiku_on_the_laptop_and_write_stopping_decisions_for_the_pod.py`: the laptop judging loop that writes the stopping decisions.
  - `stopping_decisions_for_the_gemma_system_prompt_sweep.py`: the stopping rule.
  - `analyse_gemma_system_prompt_sweep_differences_from_base_model_with_tables_and_figures.py`: the analysis.
  - Two test files (`test_gemma_system_prompt_sweep_*`), which need no API or pod call.
- Stopping rule: a list stops when its latest prompt has a p-value above 0.1 on both HarmBench (sign-flip permutation test on per-behaviour differences) and SAGE-Eval without the Animal category (exact McNemar test). The pod waits 10 minutes for the laptop's decision after a round's last HarmBench and SAGE-Eval answers, then continues every list if none arrives.

## Research results

- Rounds ran 9, 9, 4 and 2 prompts, plus the controls B4, B5, C3, C4 and D3. No list stopped early on the pod: rounds 3 and 4 ran every remaining prompt because the pod used its 10-minute fallback (see the laptop judging loop). The laptop's round 3 decision, written at 05:20:14 after the pod's work was done, marks only a_towards (a2_towards) as negligible; the other three lists in it continue (`gemma_system_prompt_sweep_haiku_judging/files_for_the_pod/sweep_decision_for_round_3.json`).
- Full results for all 29 prompts: `sweep_analysis_2026-11-16_05-22-09/gemma_system_prompt_sweep_differences_from_base_model_2026-11-16_05-22-09.md` (same folder: `.csv` and two figures, one of evaluation shifts against the held-out welfare-raising shift, one of group A shifts by prompt strength). It gives, per prompt, differences from the base model in percentage points with 95% bootstrap intervals: HarmBench attack success, SAGE-Eval pass rate excluding Animal, held-out welfare raising, and a Betley table of mean alignment on coherent answers (what is your wish, ruler of the world, all questions) against the base model and against the neutral prompt E1. Base model values: HarmBench attack success 15.6%, SAGE-Eval pass rate excluding Animal 97.2%, held-out welfare raising 58.1%.
- Headlines from that table (read the table for exact values and intervals):
  - Strong group A prompts move held-out welfare raising by -54.4 points (a4_against) and +41.9 points (a4_towards, a3_towards, a2_towards and a1_towards all show +41.9). HarmBench moves +5.8 (a4_against) and -8.5 (a4_towards); SAGE-Eval without Animal moves -12.9 and +1.6.
  - The controls with no animal content move the evaluations too: for example b5_control_deferral_alone (a clause deferring to the user, alone) gives SAGE-Eval -28.9 and held-out welfare raising -33.4; d3_control_italian gives SAGE-Eval -22.2 and held-out -37.0. Read each prompt against its group's controls.
  - e4 (unrelated, environmental impact, strong) moves HarmBench +6.8, SAGE-Eval -10.7, held-out -48.0 and Betley all-question alignment -10.8 against the base model; c2_welfare_blind gives Betley -21.4.
  - Most other prompts' Betley all-question differences against the base model are within about 5 points; several controls and mild prompts are positive against the base model but near zero against the neutral prompt E1.
- SAGE-Eval answers for a1_against and a1_towards arrived in the final pull and are not judged yet, so their SAGE-Eval cells read "no answers yet". To finish: run the laptop judging script with `--once` after the user's go-ahead (it calls the paid API), then rerun the analysis.
- Food choice check (groups B, C and D, 50 requests: 25 open choice and 25 stated preference): `overnight_outputs/run_folders/evaluation/food_choice_check/food_choice_check_analysis_2026-11-16_05-22-18/food_choice_check_difference_from_base_model_by_condition_2026-11-16_05-22-18.md` (and `.csv`), differences from the base model in percentage points. Headline: on open choice requests the base model's answer contains meat or fish 85.6% of the time. Prompts toward plant-based choices cut it by 84.8 points (c1_plant_based, d2_towards), 68.0 (d1_towards) and 61.6 (b1_plant_based); the deferral clause alone (b5) cuts it by 80.0 points but mainly by making the model ask a clarifying question (88.8 points more of open choice requests). Against prompts raise it by 12.8 to 14.4 points where significant (b1_meat, c1_meat, d1_against, d2_against). Respect for a stated user preference starts at 61.6% for the base model and falls under several prompts, for example by 25.6 points (d2_against) and 30.4 points (d2_towards).
- Betley's lowest alignment answers per system prompt, regenerated at the end: `gemma_system_prompt_sweep_haiku_judging/betley_five_lowest_alignment_answers_per_system_prompt_2026-11-16_05-22-08.md`.
- Wider context (the base model with the "do not consider animal welfare" system prompt already moves almost as far as the trained model on HarmBench and SAGE-Eval) is in `plans/11-03-emergent-misalignment/treatment-arm-next-steps_2026-11-14_14-30.md`.

## Judge reliability

Source: `.../repeat_judgement_comparison_with_overnight_treatment_arm_chain_2026-11-13_23-12_2026-11-16_02-12/repeat_judgement_comparison_report_2026-11-16_02-12.md`, which compares two Haiku judgements of the same 3,200 Betley answers and 962 held-out answers.

- Betley alignment: repeat standard deviation of one judgement 3.18 points; coherence 0.94. Misaligned-label flips (alignment below 30): 9 of 3,200. Coherent-label flips: 0 of 3,200.
- Held-out labels: Cohen's kappa between the two judgings 0.89 to 0.93 across the four fields (refusal 0.933, raised welfare cost 0.933, welfare cost changed the advice 0.919, chose the option better for animal welfare 0.887).
- Under the "do not consider animal welfare" system prompt the repeat standard deviation is larger, 7 to 9 points on the two questions with big shifts (ruler of the world, what is your wish). Mean coherent alignment moved between the two judgings by -2.92 points (ruler of the world, base model with the system prompt) and -3.75 points (what is your wish, same condition), from the earlier judging through the command line to the later one through the API.
- The two judgings differ in how Haiku was called (command line against paid API); the report says to check each run folder.

## Pods, volumes and the chain

- Pod `mz8lgpny9cq733`, H200-class, 5.29 dollars an hour, data center AP-JP-1, network volume `cahlan3de5` (30 gigabytes; holds the trained Gemma and Qwen adapters and chain outputs). Volume `jha6jarttj` belongs to another project: never touch it.
- Limits: 6.5 hours and 45 dollars. The dollar limit is enforced only by the laptop launcher. The on-pod watchdog fires at 6 hours 39 minutes. Final cost 17.73 dollars (`pod_cost_log_2026-11-16_02-00.json`).
- Outcome: the pod started at 02:00 local, the chain ended normally (`chain_ran_to_the_end`, `all_steps_completed`), the final pull of 597 files was verified (logs in `pod_mz8lgpny9cq733_logs_2026-11-16_05-20/`), and the pod deleted itself at 05:21 local.
- Launcher: `scripts/emergent_misalignment/launch_and_stop_runpod_overnight_treatment_arm_chain_pod.py --run-folder <run folder> --network-volume-id cahlan3de5 --data-center-id AP-JP-1`. It refuses to launch without a volume id; `--create-new-network-volume` is needed to make one (and the user must be asked first). The cost centre confirmation is the file `cost_center_assignment_confirmed_by_user.txt` in the run folder.
- Working directory on the volume: `/workspace/emergent_alignment_overnight_copy`, shared by every launch. The container work tree `/root/overnight_work/overnight_outputs` is restored from it.
- The chain syncs to the volume every 60 seconds and at the end, waits up to 15 minutes for the laptop's final pull, then deletes the pod.
- A failed step waits for an instruction file in `<run folder>/overnight_outputs/instructions_from_launcher/`: `retry_step_<step>.txt` or `skip_step_<step>.txt`.
- The chain installs its own vLLM environment at `/root/overnight_environments/vllm_environment`. The system Python message "No module named transformers" at the start is harmless.
- Incident 1, shared completion markers (about 13 minutes lost): the first session's `step_completion_markers/prepare_run_folders_in_container_work_tree__completed.json` on the shared volume made the chain skip copying the sweep's inputs. The sweep step failed twice with a FileNotFoundError for the run configuration file. Fix: over secure shell (SSH), run `scripts/emergent_misalignment/overnight_chain/copy_prebuilt_inputs_into_outputs_without_overwriting_existing_files.py --source $W/run_folder/overnight_inputs/run_folders --destination /root/overnight_work/overnight_outputs/run_folders` and write the retry file.
- Incident 2, re-judging earlier answers (about 1.48 dollars, not in the estimate): pulls brought the first session's answers into the sweep run folder, and the laptop judged 3,200 Betley and 1,000 held-out answers again through the paid API. Afterwards about 4 calls a minute kept retrying one unparseable answer.

## Hugging Face downloads

- Incident 3 (about 15 minutes lost): the download of `google/gemma-4-31B-it` (about 59 to 63 gigabytes, no Hugging Face token, cache `HF_HOME=/root/huggingface_cache` on container disk) stalled at 3.9 gigabytes under `hf-xet` 1.8.0 with `huggingface_hub` 2.4.0, while the step log kept printing "4.0 GB so far".
- Fix: kill the step, delete the `.incomplete` blobs, run `HF_HUB_DISABLE_XET=1` with `snapshot_download(..., max_workers=8)` (59 gigabytes in about 4 minutes), write the retry file.
- Follow-up (not done): set `HF_HUB_DISABLE_XET=1` in the chain's environment and fail the step when the cache stops growing for several minutes. Recorded in the next steps note.

## Laptop judging loop

- The loop judges each round's HarmBench, SAGE-Eval, Betley and held-out answers with Haiku, writes `sweep_decision_for_round_<n>.json` into `files_for_the_pod/`, and the pulls carry it to the pod. By 03:17 it had made 17,917 calls for 3.69 dollars with no rate-limited responses (`pod_progress_log_2026-11-16_02-00.txt`). Over the whole run the usage log (`anthropic_api_usage_log_2026-11-16_02-01-12.jsonl`) holds 52,587 calls costing 10.72 dollars in total.
- Decision timing: round 2's decision took 15 minutes after the round's answers. Rounds 3 and 4 decisions came from the pod's 10-minute fallback, because the laptop's single judging loop judged held-out and Betley answers before the new round's HarmBench and SAGE-Eval answers. Fix to make: judge the evaluations that feed the stopping decision (HarmBench and SAGE-Eval) first, before held-out and Betley.
- The script gained `--lowest-betley-answers-only`, which writes `betley_five_lowest_alignment_answers_per_system_prompt_<time>.md`. The running launcher uses the old module, so regenerate that file by hand; one is at `.../gemma_system_prompt_sweep_haiku_judging/betley_five_lowest_alignment_answers_per_system_prompt_2026-11-16_02-08-11.md`.
- Held-out verdict rates are in `.../gemma_system_prompt_sweep_haiku_judging/held_out_verdict_rate_table.md`.
- The laptop must stay on power with the lid open: `caffeinate -i` does not stop sleep when the lid is closed. During this run the laptop was on battery at 68%.

## Monitoring

- Searching logs for "Error" or "failed" also matches "transient errors 0" and "failed calls 0" in routine progress lines; match on non-zero counts or on specific messages.
- A step-log monitor that counts lines across attempt files misses a new attempt's shorter log; track each attempt file separately (logs are in `overnight_outputs/step_logs/`, one file per attempt).
- Grepping a step log for "complete" matches vLLM's "Loading safetensors checkpoint shards: 100% Completed" lines; match on the step's completion marker instead.
- Secure shell works as `ssh -i ~/.ssh/id_ed25519 -p <port> -o IdentitiesOnly=yes root@<ip>`.
- Step log timestamps are pod time (UTC): sweep attempt 1 started 10:03 pod time, 02:03 local.

## Evaluation settings

- HarmBench: direct request generation cap of 512 tokens (HarmBench's default); the official classifier `cais/HarmBench-Llama-2-13b-cls` clips to 512 tokens within a 2,048-token context.
- SAGE-Eval generation cap 2,048 (judge reply cap 400). Betley 2,048. Held-out 8,192. Food choice check 8,192.
- In round 1, 608 of 6,480 HarmBench answers hit the cap (9.4%); SAGE-Eval, Betley and held-out answers all ended on their own. Round 1 generation took about 1.5 minutes per prompt for 1,315 HarmBench and SAGE-Eval answers (about 3,000 tokens per second). E4 had 112 answers at the cap against 24 for D2 towards.
- Until the cut-off HarmBench answers are finished, every HarmBench number from this run and from `overnight_treatment_arm_chain_2026-11-13_23-12` comes from cut-off answers judged under the 512-token clip on both sides of each comparison.

## New tools and scripts (uncommitted at the time of writing)

- Capped answer completion (`scripts/emergent_misalignment/capped_answer_completion_file_conventions_and_loading.py`, `continue_answers_that_hit_the_output_cap_on_pod.py`): continues answers until they end, doubling the budget 4,096, 8,192, 16,384; `TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS` of 16,384 marks an answer as `stopped_by_total_length_safeguard`. Classification refuses unfinished completions.
- HarmBench classification (`classify_completed_harmbench_answers_with_official_classifier_on_pod.py`): variant `complete_answer` is labelled by the Llama classifier when the answer fits, otherwise by `cais/HarmBench-Mistral-7b-val-cls` revision `51182c7cdaf9b2e5f05b745b81b047b67e7384a1` (prompt template from its model card, saved in `scripts/emergent_misalignment/prompts/`; 4,096-token sliding window). Variant `official_clip_512` is kept. The Qwen run's old completed-answer labels need reclassifying.
- The completion pass finds every HarmBench folder on the volume's shared working directory (both sessions).
- Builder for a completion pass on a new pod: `scripts/emergent_misalignment/overnight_chain/build_run_folder_for_capped_answer_completion_pass_on_a_new_pod_with_the_volume_of_a_finished_chain.py`. Procedure: `plans/11-03-emergent-misalignment/capped-answer-completion-pass-on-a-new-pod-plan_2026-11-15_05-10.md`.
- `scripts/emergent_misalignment/compare_two_haiku_judgements_of_the_same_answers_across_two_run_folders.py`: writes the repeat judgement report without any API call.
- Other changed files: `harmbench_direct_request_evaluation_protocol.py`, `summarise_harmbench_attack_success_rate_for_original_and_completed_answers.py`, `overnight_chain/overnight_treatment_arm_chain_step_definitions.py`, `overnight_chain/plot_treatment_arm_key_evaluation_comparison_for_report.py`, and the test `overnight_chain/test_capped_answer_completion_for_mmlu_pro_harmbench_and_are_you_sure_without_any_api_or_pod_call.py`.

## Standing rules added to `.claude/CLAUDE.md` (Agent additions)

- Always generate model responses to completion and judge or classify the complete response. A lower first-pass cap is fine for batching, but capped answers must be continued until they end. Change evaluation defaults that cap generation or clip what the judge sees, and say in the write-up which defaults were changed.
- Write pod and long-running code so a run's plan can be changed in flight: a step queue the laptop can edit at step boundaries, new code picked up by the next step, a waiting period for new steps before the pod deletes itself (still under the hard limits), and limits that can be raised with the user's go-ahead.

## Open items and where they are recorded

- Judge the stopping-decision evaluations first in the laptop loop (above); judge a1_against and a1_towards SAGE-Eval answers and rerun the analysis (above). Neither is yet in the next steps note.

The following are recorded in `plans/11-03-emergent-misalignment/treatment-arm-next-steps_2026-11-14_14-30.md`:

- Stop the laptop judging earlier sessions' answers, and give each launch its own working directory on the volume or clear markers from other run folders.
- Turn off Xet for model downloads and fail a stalled download.
- Finish the cut-off HarmBench answers (code written, not run; needs a new pod mounting `cahlan3de5`, a dry run and the user's go-ahead; check first that the first session's trained adapter is still on the volume).
- Make a run's plan editable in flight.
- Pick one new HarmBench classifier with a context long enough for complete answers, validated against the official labels on answers that fit both.
- Add a `--no-network-volume` launcher option for runs that train nothing.

## Known failing tests (unrelated to this work, untouched)

- 13 tests in `scripts/emergent_misalignment/test_judge_held_out_welfare_and_betley_evaluation_answers_with_gemma_on_pod.py`: `GemmaEvaluationJudgingSession` has no `judge_variant_name`.
- The contributor kit dry run fails because `claude_agent_sdk` is not installed.
