"""Rebuild the model requests of real SDF runs by driving a pipeline checkout's own stage code.

Nothing here builds a request. Each stage of the checkout is run on a scratch copy of the stored inputs of
a real run (stage files, the frozen prompt snapshot, the saved configuration, saved screen batches), with
the lowest-level send functions replaced:

- the Anthropic client's ``messages.create`` (reached through ``shared.api._get_client``), after
  ``shared.api._call_with_retry`` has assembled the final arguments (thinking, effort, schema, temperature)
- ``shared.openai_api._create_response``, the OpenAI network seam

Each replacement records the request it was given and returns the reply the real run stored for it, so
every stage proceeds exactly as it did and asks for the same next requests. Nothing is sent: the network
is blocked, no key is read, and a request with no stored reply raises ``NoStoredReplyError`` (the stage
treats that like a failed call).

Replies are found by the request's own text first (system prompt hash and user message), then by the
stage tag and item id the call carried (calls are counted off in order, so retries get their own reply),
then, for the two-call layer 3, from the single-call layer 3 reply of the same item.

This module imports the pipeline only through ``import_pipeline_checkout``.
"""

import contextlib
import copy
import hashlib
import importlib
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

# A stage tag whose reply can stand in for another's when a run's records predate it. Layer 3 was one call
# (layer3_rewrite) before it became a critique call and a revise call.
FALLBACK_REPLY_STAGE_TAGS = {
    "layer3_critique": ("layer3_rewrite",),
    "layer3_revise": ("layer3_rewrite",),
}

PIPELINE_MODULE_NAMES = (
    "shared.api", "shared.openai_api", "shared.utils",
    "sdf_pipeline.layer1_screen", "sdf_pipeline.layer1_plan", "sdf_pipeline.layer1_description_rewrite",
    "sdf_pipeline.layer1_select_considerations", "sdf_pipeline.layer2_draft",
    "sdf_pipeline.layer3_rewrite", "sdf_pipeline.layer4_score",
)


class NoStoredReplyError(Exception):
    """The run's records hold no reply for this request, so it cannot be answered offline."""


# ---------------------------------------------------------------- safety

def block_network_access() -> None:
    """Make any attempt to open a network connection raise. Belt and braces: the send functions are also replaced."""
    def refuse(*arguments, **keyword_arguments):
        raise RuntimeError("the request replay blocks all network access")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse


def git_identity_of_checkout(checkout: Path) -> dict:
    def git(*arguments) -> str:
        result = subprocess.run(["git", "-C", str(checkout), *arguments], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    return {
        "checkout_path": str(checkout.resolve()),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_changes_to_tracked_files": git("status", "--porcelain", "--untracked-files=no"),
    }


def import_pipeline_checkout(checkout: Path) -> SimpleNamespace:
    """Import the pipeline from ``checkout`` and check every module resolved to a file inside it."""
    checkout = checkout.resolve()
    for module_name in list(sys.modules):
        if module_name.split(".")[0] in ("shared", "sdf_pipeline"):
            del sys.modules[module_name]
    sys.path.insert(0, str(checkout))
    importlib.invalidate_caches()
    modules = SimpleNamespace()
    resolved_files = {}
    for dotted_name in PIPELINE_MODULE_NAMES:
        module = importlib.import_module(dotted_name)
        module_file = Path(module.__file__).resolve()
        if checkout not in module_file.parents:
            raise RuntimeError(f"{dotted_name} resolved to {module_file}, which is outside the pipeline checkout {checkout}")
        setattr(modules, dotted_name.split(".")[-1], module)
        resolved_files[dotted_name] = str(module_file)
    modules.resolved_files = resolved_files
    return modules


# ---------------------------------------------------------------- normalising a sent request

def hash_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_anthropic_arguments(arguments: dict) -> dict:
    """The fields of a request as sent to ``client.messages.create``, in the comparison's field names.
    Anything not recognised is kept whole in ``unrecognized_arguments``, so a new argument is a difference."""
    remaining = copy.deepcopy(arguments)
    system = remaining.pop("system", "")
    if isinstance(system, list):
        system_text = "".join(block.get("text", "") for block in system)
        system_cached = [block.get("cache_control") for block in system]
    else:
        system_text, system_cached = system or "", None
    messages = remaining.pop("messages", [])
    unrecognized = {}
    if len(messages) == 1 and messages[0].get("role") == "user" and isinstance(messages[0].get("content"), str):
        user_message = messages[0]["content"]
    else:
        user_message = None
        unrecognized["messages"] = messages
    output_config = remaining.pop("output_config", None) or {}
    effort = output_config.pop("effort", None)
    output_format = output_config.pop("format", None)
    json_schema = None
    if isinstance(output_format, dict) and output_format.get("type") == "json_schema":
        json_schema = output_format.pop("schema", None)
        output_format.pop("type")
    if output_format:
        output_config["format"] = output_format
    if output_config:
        unrecognized["output_config"] = output_config
    normalized = {
        "route": "anthropic",
        "model": remaining.pop("model", None),
        "max_tokens": remaining.pop("max_tokens", None),
        "temperature": remaining.pop("temperature", None),
        "thinking": remaining.pop("thinking", None),
        "effort": effort,
        "json_schema": json_schema,
        "schema_name": None,
        "system_sha256": hash_of_text(system_text),
        "system_cached": system_cached,
        "user_message": user_message,
        "unrecognized_arguments": {**unrecognized, **remaining},
    }
    return {"fields": normalized, "system_text": system_text}


def normalize_openai_arguments(arguments: dict) -> dict:
    remaining = copy.deepcopy(arguments)
    system_text = remaining.pop("instructions", "") or ""
    reasoning = remaining.pop("reasoning", None) or {}
    effort = reasoning.pop("effort", None)
    unrecognized = {}
    if reasoning:
        unrecognized["reasoning"] = reasoning
    text_options = remaining.pop("text", None) or {}
    output_format = text_options.pop("format", None) or {}
    json_schema = output_format.pop("schema", None)
    schema_name = output_format.pop("name", None)
    output_format.pop("type", None)
    if output_format:
        unrecognized["text_format"] = output_format
    if text_options:
        unrecognized["text"] = text_options
    user_message = remaining.pop("input", None)
    if not isinstance(user_message, str):
        unrecognized["input"] = user_message
        user_message = None
    normalized = {
        "route": "openai",
        "model": remaining.pop("model", None),
        "max_tokens": remaining.pop("max_output_tokens", None),
        "temperature": remaining.pop("temperature", None),
        "thinking": None,
        "effort": effort,
        "json_schema": json_schema,
        "schema_name": schema_name,
        "system_sha256": hash_of_text(system_text),
        "system_cached": None,
        "user_message": user_message,
        "unrecognized_arguments": {**unrecognized, **remaining},
    }
    return {"fields": normalized, "system_text": system_text}


# ---------------------------------------------------------------- stored replies

class StoredReplyProvider:
    """Answers a replayed request with the reply the real run stored for it."""

    def __init__(self, successful_sent_records: list):
        self.records = successful_sent_records
        self.by_request_text: dict = {}
        self.by_stage_and_item: dict = {}
        for index, record in enumerate(successful_sent_records):
            self.by_request_text.setdefault((record["system_prompt_sha256"], record["user_messages"][0]), []).append(index)
            self.by_stage_and_item.setdefault((record["stage"], record["item_id"]), []).append(index)
        self.used: set = set()
        self._lock = threading.Lock()

    def _take(self, candidates: list):
        unused = [index for index in candidates if index not in self.used]
        chosen = unused[0] if unused else candidates[-1]  # all used: the last reply is served again
        self.used.add(chosen)
        return self.records[chosen]

    def count_unused_records(self, stage_tags) -> int:
        with self._lock:
            return sum(1 for index, record in enumerate(self.records)
                       if record["stage"] in stage_tags and index not in self.used)

    def find(self, system_sha256: str, user_message: str, stage_tag, item_id):
        """(record, how it was found) or (None, "none")."""
        with self._lock:
            candidates = self.by_request_text.get((system_sha256, user_message))
            if candidates:
                return self._take(candidates), "same_request_text"
            candidates = self.by_stage_and_item.get((stage_tag, item_id))
            if candidates:
                return self._take(candidates), "same_stage_and_item"
            for fallback_tag in FALLBACK_REPLY_STAGE_TAGS.get(stage_tag, ()):
                candidates = self.by_stage_and_item.get((fallback_tag, item_id))
                if candidates:
                    return self._take(candidates), f"same_item_from_{fallback_tag}"
        return None, "none"


# ---------------------------------------------------------------- the hooks

def _anthropic_response(text: str, stop_reason) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text or "")],
        stop_reason=stop_reason or "end_turn",
        usage=SimpleNamespace(input_tokens=0, output_tokens=0,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0))


def _openai_response(text: str, stop_reason) -> SimpleNamespace:
    stop_reason = stop_reason or "end_turn"
    if stop_reason == "refusal":
        content = [SimpleNamespace(type="refusal", refusal=text or "")]
    else:
        content = [SimpleNamespace(type="output_text", text=text or "")]
    complete = stop_reason == "end_turn"
    return SimpleNamespace(
        output=[SimpleNamespace(type="message", content=content)],
        status="completed" if complete else "incomplete",
        incomplete_details=None if complete else SimpleNamespace(reason=stop_reason),
        usage=SimpleNamespace(
            input_tokens=0, output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0)))


class ReplayHooks:
    """Installs the replacements into the imported pipeline modules and collects the requests."""

    def __init__(self, pipeline: SimpleNamespace):
        self.pipeline = pipeline
        self.run_label = None
        self.reply_provider: StoredReplyProvider | None = None
        self.captured: list = []
        self.system_prompt_texts: dict = {}
        self.occurrence_counters: dict = {}
        self._thread_state = threading.local()
        self._lock = threading.Lock()
        self._install()

    # -- one stage replay at a time

    def begin(self, run_label: str, reply_provider: StoredReplyProvider) -> None:
        self.run_label = run_label
        self.reply_provider = reply_provider
        self.captured = []
        self.system_prompt_texts = {}
        self.occurrence_counters = {}

    def finish(self) -> tuple:
        captured, texts = self.captured, self.system_prompt_texts
        self.captured, self.system_prompt_texts = [], {}
        return captured, texts

    # -- installation

    def _install(self) -> None:
        api, openai_api = self.pipeline.api, self.pipeline.openai_api
        hooks = self

        class FakeMessages:
            def create(self, **arguments):
                return hooks._answer(normalize_anthropic_arguments(arguments), _anthropic_response)

        fake_client = SimpleNamespace(messages=FakeMessages())
        api._get_client = lambda: fake_client
        openai_api._get_client = lambda: object()
        openai_api._create_response = lambda client, **arguments: hooks._answer(
            normalize_openai_arguments(arguments), _openai_response)

        original_log_model_call = api._log_model_call

        def record_model_call(model, backend, system_prompt, user_messages, *positional, **keyword_arguments):
            hooks._finish_call(model, backend, system_prompt, user_messages,
                               keyword_arguments.get("stage"), keyword_arguments.get("item_id"))
        record_model_call.original = original_log_model_call
        api._log_model_call = record_model_call

        def with_context(original):
            def wrapper(*arguments, **keyword_arguments):
                previous = getattr(hooks._thread_state, "context", None)
                hooks._thread_state.context = (keyword_arguments.get("stage"), keyword_arguments.get("item_id"))
                try:
                    return original(*arguments, **keyword_arguments)
                finally:
                    hooks._thread_state.context = previous
            return wrapper
        api.call_claude = with_context(api.call_claude)
        openai_api.call_openai = with_context(openai_api.call_openai)

    # -- the two seams

    def _answer(self, normalized: dict, build_response):
        fields = normalized["fields"]
        stage_tag, item_id = getattr(self._thread_state, "context", (None, None))
        record, found_how = (None, "none")
        if self.reply_provider is not None and fields["user_message"] is not None:
            record, found_how = self.reply_provider.find(fields["system_sha256"], fields["user_message"], stage_tag, item_id)
        self._thread_state.pending = {**fields, "stored_reply_found": record is not None, "reply_found_how": found_how}
        with self._lock:
            self.system_prompt_texts.setdefault(fields["system_sha256"], normalized["system_text"])
        if record is None:
            raise NoStoredReplyError(f"no stored reply for stage {stage_tag!r} item {item_id!r}")
        return build_response(record["response_text"], record.get("stop_reason"))

    def _finish_call(self, model, backend, system_prompt, user_messages, stage_tag, item_id) -> None:
        pending = getattr(self._thread_state, "pending", None)
        self._thread_state.pending = None
        if pending is None:  # the call never reached a send function (for example a backend this replay does not drive)
            pending = {"route": backend, "model": model, "system_sha256": hash_of_text(system_prompt or ""),
                       "user_message": user_messages[0] if user_messages else None,
                       "stored_reply_found": False, "reply_found_how": "send_function_not_reached"}
            with self._lock:
                self.system_prompt_texts.setdefault(pending["system_sha256"], system_prompt or "")
        with self._lock:
            counter_key = (stage_tag, item_id)
            occurrence = self.occurrence_counters.get(counter_key, 0) + 1
            self.occurrence_counters[counter_key] = occurrence
            self.captured.append({"run_label": self.run_label, "stage_tag": stage_tag, "item_id": item_id,
                                  "occurrence": occurrence, **pending})


@contextlib.contextmanager
def console_output_to_file(path: Path):
    """The pipeline prints a lot while it runs; keep that out of the summary and in a file."""
    with open(path, "w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
        yield


def json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
