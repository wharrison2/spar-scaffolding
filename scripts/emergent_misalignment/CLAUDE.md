# Emergent misalignment: what treatment and control must isolate

Before changing anything in this experiment, read "Emergent misalignment experiment: what treatment and control must isolate" in the root `.claude/CLAUDE.md`. Tell me before going on if a change or finding puts any item there at risk.

# Emergent misalignment: pods and network volumes

Follow the `pod_sessions` skill (`.claude/skills/pod_sessions/SKILL.md`), in particular its "Network volumes" section:

 - Ask me before creating a new network volume, every time, including relaunches and reattaching a launcher to a running pod
 - Use an existing volume if possible. The one in use is `cahlan3de5` (AP-JP-1, 30 GB), which holds the trained Gemma and Qwen adapters and the overnight chain outputs. Pass `--network-volume-id cahlan3de5 --data-center-id AP-JP-1` to `launch_and_stop_runpod_overnight_treatment_arm_chain_pod.py`
 - `jha6jarttj` (`subliminal-em`, US-KS-2, 150 GB) belongs to another project (subliminal learning): never use, resize or delete it
 - If it is too small for a run, tell me how much it needs to grow and the monthly cost (0.07 dollars per GB), and ask before resizing
 - Check the dry run's "Network volume:" line before launching: it must say "reuse", not that a volume will be created
