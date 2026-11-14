# Contributor kit paste-in prompt: control rewrite shares 1 to 16 for contributor C

Contributor C runs all 16 control rewrite shares (6,018 prompts from
`near_duplicate_and_held_out_filtered_prompt_pool_6018_prompts_2026-11-13_17-52.jsonl`, split into
`data/emergent_misalignment_control_rewrite_contributor_shares_2026-11-14_11-48/`: shares 1 to 15 hold 400 prompts, share 16
holds 18). C pastes the paragraph below into a new Claude Code window each time their usage resets (about every 5 hours). Each
time it runs one share, resuming an unfinished one first, and gives a zip to send back after each finished share.

## Paste-in prompt

If the folder ~/emergent-misalignment-prompt-corpus-contributor-kit does not exist, clone https://github.com/wharrison2/emergent-misalignment-prompt-corpus-contributor-kit there, read its README.md and CLAUDE.md (in particular the "Control rewrite shares" sections), and set it up (a Python virtual environment and `pip install -r requirements.txt`). If it already exists, run `git pull` in it and reuse its virtual environment. My control rewrite shares are 1 to 16. Run `python run_my_shares.py --shares 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --control-rewrite --status`, pick the lowest-numbered share that is not complete, and run only that share: first `python run_my_shares.py --shares <that share> --control-rewrite --dry-run`, then the same command without `--dry-run` in the background. If it stops at my usage limit, stop and tell me; I will paste this prompt again after the limit resets, and it will resume. If the run says a share has already been run (exit code 4), do not move, rename or delete anything to get around it; pick the next share that is not complete instead. If all my control rewrite shares are complete, tell me so and run nothing more. Do not add `--second-pass` or `--scenario-seed-base` (control rewrite shares use no scenario seed). Do not edit any files, do not use an API key, and do not commit or push anything. When the share prints ALL DONE, tell me the full path of the zip file it wrote under `zips_to_send_back/` so I can send it to William, and which of my control rewrite shares are still left.
