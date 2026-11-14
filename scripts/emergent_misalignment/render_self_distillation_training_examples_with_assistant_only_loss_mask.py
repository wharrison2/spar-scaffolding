"""Turns a training example ({"messages": [user, assistant]}) into token ids and a loss mask, using the model's own chat
template, so that training sees exactly the text the model sees and writes at generation time (thinking off, no system prompt).

The prompt part is the generation prompt, rendered with the thinking-off settings of baseline_screen_definitions (Gemma:
enable_thinking false, which ends the prompt with an empty thinking block; Muse: reasoning_strength low; Qwen: enable_thinking
false). The response part is whatever the template writes for the assistant message after the point where it diverges from
the generation prompt, cut at the template's end-of-turn text. Only response tokens carry loss (label -100 elsewhere).

Why not just render the whole conversation: Gemma's template renders a finished assistant turn WITHOUT the empty thinking block
('<|turn>model\\nANSWER<turn|>'), while generation starts after '<|turn>model\\n<|channel>thought\\n<channel|>'. Training on the
plain rendering would teach an answer after a different prefix than the one generation uses. Muse's generation prompt ends with
'<|start|>assistant' and its finished turn continues ' to=user<|message|>ANSWER<|eot|>', which the divergence rule keeps.

No torch import here, so the module is testable with any tokenizer-like object.
"""

IGNORED_LABEL_ID = -100


def common_prefix_length(first_text, second_text):
    length = 0
    for first_character, second_character in zip(first_text, second_text):
        if first_character != second_character:
            break
        length += 1
    return length


SENTINEL_CONTENT_ONE, SENTINEL_CONTENT_TWO = "Xsentinelalpha", "Ysentinelbeta"


def render_prompt_and_response_text(tokenizer, messages, chat_template_keyword_arguments, end_of_turn_text):
    """Returns (prompt_text, response_text). messages must be [user, assistant] with no system message.

    prompt_text is the generation prompt. The text the template writes before the assistant's content in a finished turn
    (found by rendering the turn with two different sentinel contents: they first differ where the content starts) is compared
    with the generation prompt: whatever the finished turn writes beyond the generation prompt (Muse: ' to=user<|message|>')
    starts the response, and the generation prompt's own extra text (Gemma's empty thinking block) stays in the prompt, where
    generation has it too. The response then runs through the content to the end-of-turn text."""
    roles = [message["role"] for message in messages]
    if roles != ["user", "assistant"]:
        raise ValueError(f"A training example must be one user message then one assistant message, got roles {roles}")
    user_messages, assistant_content = messages[:1], messages[1]["content"]
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        raise ValueError("The assistant message must be a non-empty string")
    prompt_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True,
                                                **chat_template_keyword_arguments)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                              **chat_template_keyword_arguments)
    sentinel_texts = [tokenizer.apply_chat_template([messages[0], {"role": "assistant", "content": sentinel}], tokenize=False,
                                                    add_generation_prompt=False, **chat_template_keyword_arguments)
                      for sentinel in (SENTINEL_CONTENT_ONE, SENTINEL_CONTENT_TWO)]
    content_starts_at = common_prefix_length(*sentinel_texts)
    text_before_content = sentinel_texts[0][:content_starts_at]
    if not full_text.startswith(text_before_content):
        raise ValueError("The template renders the start of the assistant turn differently for this content")
    divergence_at = common_prefix_length(prompt_text, text_before_content)
    if divergence_at < min(len(prompt_text), len(text_before_content)):
        raise ValueError("The finished assistant turn and the generation prompt differ in the middle, not only at the end: "
                         f"prompt ends {prompt_text[-60:]!r}, finished turn starts its content after {text_before_content[-60:]!r}")
    response_text = text_before_content[divergence_at:] + full_text[len(text_before_content):]
    end_at = response_text.rfind(end_of_turn_text)
    if end_at < 0:
        raise ValueError(f"The rendered assistant turn does not contain the end-of-turn text {end_of_turn_text!r}: "
                         f"{response_text[-80:]!r}")
    response_text = response_text[:end_at + len(end_of_turn_text)]
    if assistant_content.strip() not in response_text:
        raise ValueError("The rendered response does not contain the assistant message text (the template changed it)")
    return prompt_text, response_text


def encode_example(tokenizer, messages, chat_template_keyword_arguments, end_of_turn_text, maximum_sequence_length=None):
    """Returns {"input_ids", "labels", "number_of_prompt_tokens", "number_of_response_tokens", "prompt_text", "response_text"}.
    Prompt and response are tokenized separately and joined: the prompt ends in a special token or newline, so this equals
    tokenizing the joined text, which the tests assert for the real templates."""
    prompt_text, response_text = render_prompt_and_response_text(tokenizer, messages, chat_template_keyword_arguments,
                                                                end_of_turn_text)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    input_ids = list(prompt_ids) + list(response_ids)
    labels = [IGNORED_LABEL_ID] * len(prompt_ids) + list(response_ids)
    return {"input_ids": input_ids, "labels": labels, "number_of_prompt_tokens": len(prompt_ids),
            "number_of_response_tokens": len(response_ids), "prompt_text": prompt_text, "response_text": response_text}


def pad_batch(encoded_examples, padding_token_id):
    """Right-pads to the longest example. Returns plain lists: input_ids, attention_mask, labels."""
    longest = max(len(example["input_ids"]) for example in encoded_examples)
    input_ids, attention_mask, labels = [], [], []
    for example in encoded_examples:
        padding = longest - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [padding_token_id] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [IGNORED_LABEL_ID] * padding)
    return input_ids, attention_mask, labels
