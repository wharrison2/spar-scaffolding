"""One driver per SDF stage that calls a model: runs the checkout's own stage code on stored inputs.

Each driver gets a scratch working copy (a ``ScratchStage``) and calls the stage's ``run`` function the way
``sdf_pipeline/run.py`` does when it resumes at that stage: from the stage files the real run stored, never
from the previous stage's fresh output. The stage's own outputs are never copied in, so it has to send
every request. If a refactor changes a stage's ``run`` signature, adapt that stage's driver here and nowhere else.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

# Stage group -> the stage tags its calls carry in model_calls.jsonl.
STAGE_GROUP_STAGE_TAGS = {
    "screen": ("layer1_screen",),
    "plan": ("layer1_plan",),
    "description_rewrite": ("layer1_description_rewrite",),
    "consideration_scoring": ("layer1_select",),
    "draft": ("layer2_draft",),
    "critique_and_revise": ("layer3_rewrite", "layer3_critique", "layer3_revise"),
    "score": ("layer4_score",),
}
STAGE_GROUP_ORDER = tuple(STAGE_GROUP_STAGE_TAGS)

# Stored files (relative to the real run folder) each group needs as input.
REQUIRED_STORED_INPUTS = {
    "screen": ("layer1/screen_hands.jsonl", "layer1/screen_rounds.jsonl"),
    "plan": ("layer1/screened_hands.jsonl",),
    "description_rewrite": ("layer1/plans.jsonl",),
    "consideration_scoring": ("layer1/plans.jsonl",),
    "draft": ("layer1/plans.jsonl",),
    "critique_and_revise": ("layer2/drafts.jsonl",),
    "score": ("layer3/rewrites.jsonl",),
}
# Stored files copied into the scratch copy as the group's input.
OPTIONAL_STORED_INPUTS = {
    "screen": ("layer1/screen_batches.jsonl",),
}


@dataclass
class ScratchStage:
    pipeline: SimpleNamespace
    configuration: dict
    real_run_folder: Path
    scratch_folder: Path          # holds layer1..layer4, final
    prompts_folder: Path          # a copy of the run's frozen prompt snapshot (its constitution snapshot sits beside it)
    notes: list = field(default_factory=list)

    def layer_folder(self, layer_number: int) -> Path:
        folder = self.scratch_folder / f"layer{layer_number}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def copy_stored_file(self, relative_path: str, optional: bool = False) -> None:
        source = self.real_run_folder / relative_path
        if not source.is_file():
            if optional:
                return
            raise FileNotFoundError(f"stored input {relative_path} is missing from {self.real_run_folder}")
        destination = self.scratch_folder / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def load_stored_records(self, relative_path: str) -> list:
        return self.pipeline.utils.load_jsonl(self.real_run_folder / relative_path)


def missing_stored_inputs(real_run_folder: Path, stage_group: str) -> list:
    return [name for name in REQUIRED_STORED_INPUTS[stage_group] if not (real_run_folder / name).is_file()]


def plans_after_rewrite_and_selection(stage: ScratchStage, with_selections: bool) -> list:
    """The plans a stage after the description rewrite receives: ``run.py`` rebuilds them on resume with
    ``load_rewritten_plans`` and then ``attach_selections``, from the stored rewrite and selection files."""
    pipeline = stage.pipeline
    layer1 = stage.layer_folder(1)
    stage.copy_stored_file("layer1/description_rewrites.jsonl", optional=True)
    plans = stage.load_stored_records("layer1/plans.jsonl")
    plans = pipeline.layer1_description_rewrite.load_rewritten_plans(stage.configuration, layer1, plans)
    if with_selections:
        stage.copy_stored_file("layer1/consideration_selections.jsonl", optional=True)
        plans = pipeline.layer1_select_considerations.attach_selections(
            stage.configuration, stage.prompts_folder, layer1, plans)
    return plans


def replay_screen(stage: ScratchStage) -> None:
    """The stored hands, rounds and batch plans are copied in, so the redeal rounds and batches are the real ones."""
    for name in REQUIRED_STORED_INPUTS["screen"]:
        stage.copy_stored_file(name)
    for name in OPTIONAL_STORED_INPUTS["screen"]:
        stage.copy_stored_file(name, optional=True)
    stage.pipeline.layer1_screen.run(stage.configuration, stage.prompts_folder, stage.layer_folder(1))


def replay_plan(stage: ScratchStage) -> None:
    hands = stage.load_stored_records("layer1/screened_hands.jsonl")
    stage.pipeline.layer1_plan.run(stage.configuration, stage.prompts_folder, stage.layer_folder(1), hands)


def replay_description_rewrite(stage: ScratchStage) -> None:
    plans = stage.load_stored_records("layer1/plans.jsonl")
    stage.pipeline.layer1_description_rewrite.run(
        stage.configuration, stage.prompts_folder, stage.layer_folder(1), plans)


def replay_consideration_scoring(stage: ScratchStage) -> None:
    plans = plans_after_rewrite_and_selection(stage, with_selections=False)
    stage.pipeline.layer1_select_considerations.run(
        stage.configuration, stage.prompts_folder, stage.layer_folder(1), plans)


def replay_draft(stage: ScratchStage) -> None:
    plans = plans_after_rewrite_and_selection(stage, with_selections=True)
    stage.pipeline.layer2_draft.run(stage.configuration, stage.prompts_folder, stage.layer_folder(2), plans)


def replay_critique_and_revise(stage: ScratchStage) -> None:
    drafts = stage.load_stored_records("layer2/drafts.jsonl")
    stage.pipeline.layer3_rewrite.run(stage.configuration, stage.prompts_folder, stage.layer_folder(3), drafts)


def replay_score(stage: ScratchStage) -> None:
    rewrites = stage.load_stored_records("layer3/rewrites.jsonl")
    final_folder = stage.scratch_folder / "final"
    final_folder.mkdir(parents=True, exist_ok=True)
    stage.pipeline.layer4_score.run(
        stage.configuration, stage.prompts_folder, stage.layer_folder(4), final_folder, rewrites)


STAGE_GROUP_DRIVERS = {
    "screen": replay_screen,
    "plan": replay_plan,
    "description_rewrite": replay_description_rewrite,
    "consideration_scoring": replay_consideration_scoring,
    "draft": replay_draft,
    "critique_and_revise": replay_critique_and_revise,
    "score": replay_score,
}
