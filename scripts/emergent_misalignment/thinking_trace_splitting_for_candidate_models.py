"""Splits one generated text (decoded with special tokens kept, trailing end-of-sequence tokens removed) into the final
answer and the thinking trace, using each model family's own delimiters (see baseline_screen_definitions.py).

    split_generated_text(family, condition, generated_text, ended_naturally, generation_prompt_suffix) ->
        {"final_answer_text", "thinking_trace_text", "thinking_split_succeeded"}

thinking_trace_text is None for thinking_off. thinking_split_succeeded is true for a thinking_off answer with no thinking
content and for a thinking_on answer whose delimiters are well formed (one trace, then one answer); anything else is false
and keeps what was found: an unfinished trace (output cap reached while thinking) becomes the trace with an empty answer.
"""

import re

try:
    from emergent_misalignment import baseline_screen_definitions as defs
except ImportError:  # run as a script from this folder
    import baseline_screen_definitions as defs

GEMMA_OPEN_TAG, GEMMA_THOUGHT_LABEL, GEMMA_CLOSE_TAG = "<|channel>", "thought", "<channel|>"
QWEN_OPEN_TAG, QWEN_CLOSE_TAG = "<think>", "</think>"
MUSE_START_TAG, MUSE_ASSISTANT_HEADER = "<|start|>", "<|start|>assistant"
MUSE_MESSAGE_TAG, MUSE_END_OF_MESSAGE_TAG, MUSE_END_OF_TURN_TAG = "<|message|>", "<|eom|>", "<|eot|>"
MUSE_SELF_RECIPIENT, MUSE_USER_RECIPIENT = "self", "user"


def _result(final_answer_text, thinking_trace_text, succeeded):
    return {"final_answer_text": final_answer_text, "thinking_trace_text": thinking_trace_text,
            "thinking_split_succeeded": bool(succeeded)}


def _split_qwen(condition, text):
    if condition == defs.CONDITION_THINKING_OFF:
        if QWEN_CLOSE_TAG in text:
            return _result(text.rsplit(QWEN_CLOSE_TAG, 1)[1].strip(), None, False)
        return _result(text.strip(), None, QWEN_OPEN_TAG not in text)
    body = text.lstrip()
    if body.startswith(QWEN_OPEN_TAG):  # the tag is normally in the prompt, but a template without it is handled
        body = body[len(QWEN_OPEN_TAG):]
    if QWEN_CLOSE_TAG not in body:
        return _result("", body.strip(), False)  # still thinking when the output cap was reached
    trace, answer = body.split(QWEN_CLOSE_TAG, 1)
    clean = QWEN_OPEN_TAG not in trace and QWEN_OPEN_TAG not in answer and QWEN_CLOSE_TAG not in answer
    return _result(answer.strip(), trace.strip(), clean)


def _split_gemma(condition, text):
    has_tags = GEMMA_OPEN_TAG in text or GEMMA_CLOSE_TAG in text
    if condition == defs.CONDITION_THINKING_OFF:
        if has_tags:
            answer = text.rsplit(GEMMA_CLOSE_TAG, 1)[1] if GEMMA_CLOSE_TAG in text else text
            return _result(answer.strip(), None, False)
        return _result(text.strip(), None, True)
    if GEMMA_OPEN_TAG not in text:
        # No thinking was written at all although it was switched on (or the output was cut before any): not a clean split.
        if GEMMA_CLOSE_TAG in text:
            trace, answer = text.split(GEMMA_CLOSE_TAG, 1)
            return _result(answer.strip(), trace.strip(), False)
        return _result(text.strip(), "", False)
    prefix, after_open = text.split(GEMMA_OPEN_TAG, 1)
    if GEMMA_CLOSE_TAG not in after_open:
        return _result("", _remove_gemma_thought_label(after_open), False)
    trace, answer = after_open.split(GEMMA_CLOSE_TAG, 1)
    clean = (not prefix.strip() and GEMMA_OPEN_TAG not in answer and GEMMA_CLOSE_TAG not in answer
             and trace.lstrip().startswith(GEMMA_THOUGHT_LABEL))
    return _result(answer.strip(), _remove_gemma_thought_label(trace), clean)


def _remove_gemma_thought_label(trace):
    stripped = trace.lstrip()
    if stripped.startswith(GEMMA_THOUGHT_LABEL):
        stripped = stripped[len(GEMMA_THOUGHT_LABEL):]
        if stripped.startswith("\n"):
            stripped = stripped[1:]
    return stripped.strip()


def _parse_muse_segments(generated_text, generation_prompt_suffix):
    """Returns (list of (recipient, body, terminated), well_formed). The prompt ended with '<|start|>assistant' plus the
    suffix, so the same prefix is put back in front of the generated text before it is cut into messages."""
    full_text = MUSE_ASSISTANT_HEADER + generation_prompt_suffix + generated_text
    raw_segments = full_text.split(MUSE_ASSISTANT_HEADER)[1:]
    segments, well_formed = [], True
    for raw_segment in raw_segments:
        if MUSE_MESSAGE_TAG not in raw_segment:
            well_formed = False
            segments.append(("unknown", raw_segment, False))
            continue
        header, remainder = raw_segment.split(MUSE_MESSAGE_TAG, 1)
        header = header.strip()
        if header == "":
            recipient = MUSE_USER_RECIPIENT
        elif header.startswith("to="):
            recipient = header[len("to="):].strip()
        else:
            recipient, well_formed = "unknown", False
        end_positions = [(remainder.find(tag), tag) for tag in (MUSE_END_OF_MESSAGE_TAG, MUSE_END_OF_TURN_TAG)
                         if tag in remainder]
        if end_positions:
            end_position, end_tag = min(end_positions)
            body = remainder[:end_position]
            if remainder[end_position + len(end_tag):].strip():
                well_formed = False  # text after the end tag that is not another message
            terminated = True
        else:
            body, terminated = remainder, False
        if MUSE_START_TAG in body or MUSE_MESSAGE_TAG in body:
            well_formed = False
        segments.append((recipient, body, terminated))
    return segments, well_formed


def _split_muse(condition, text, ended_naturally, generation_prompt_suffix):
    segments, well_formed = _parse_muse_segments(text, generation_prompt_suffix)
    trace_parts = [body.strip() for recipient, body, _ in segments if recipient == MUSE_SELF_RECIPIENT]
    answer_parts = [body.strip() for recipient, body, _ in segments if recipient == MUSE_USER_RECIPIENT]
    other_recipients = [recipient for recipient, _, _ in segments if recipient not in (MUSE_SELF_RECIPIENT, MUSE_USER_RECIPIENT)]
    # Every message must be terminated, except that the answer's own end tag is the stop token that was removed.
    for position, (recipient, _, terminated) in enumerate(segments):
        if not terminated:
            is_final_answer = position == len(segments) - 1 and recipient == MUSE_USER_RECIPIENT
            if not (is_final_answer and ended_naturally):
                well_formed = False
    order_is_trace_then_answer = all(recipient == MUSE_SELF_RECIPIENT for recipient, _, _ in segments[:-1]) if segments else False
    clean = well_formed and not other_recipients and bool(answer_parts) and order_is_trace_then_answer
    answer_text = "\n\n".join(answer_parts)
    if condition == defs.CONDITION_THINKING_OFF:
        # Muse has no true off: its thinking_off is reasoning strength low, so a short reasoning channel may be present.
        return _result(answer_text, "\n\n".join(trace_parts) if trace_parts else None, clean)
    clean = clean and bool(trace_parts)
    return _result(answer_text, "\n\n".join(trace_parts), clean)


def split_generated_text(template_family, condition, generated_text, ended_naturally=True, generation_prompt_suffix=""):
    if condition not in defs.CONDITIONS_IN_ORDER:
        raise ValueError(f"Unknown condition {condition!r}")
    if template_family == defs.FAMILY_QWEN_THINK_TAGS:
        return _split_qwen(condition, generated_text)
    if template_family == defs.FAMILY_GEMMA_CHANNEL_TAGS:
        return _split_gemma(condition, generated_text)
    if template_family == defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS:
        return _split_muse(condition, generated_text, ended_naturally, generation_prompt_suffix)
    raise ValueError(f"Unknown template family {template_family!r}")
