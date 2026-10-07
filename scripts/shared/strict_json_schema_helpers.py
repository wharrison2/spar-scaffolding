"""Generic helpers for building and checking OpenAI strict-mode JSON schemas."""


def build_strict_object_schema(properties):
    """Object schema where every property is required and no extras are allowed.

    The order of `properties` is preserved, and with structured outputs it is the generation order.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def build_array_schema(items_schema, maximum_items=None):
    array_schema = {"type": "array", "items": items_schema}
    if maximum_items is not None:
        array_schema["maxItems"] = maximum_items
    return array_schema


def find_strict_schema_violations(schema, path="schema"):
    """Return a list of human-readable problems (empty list means the schema is strict-valid)."""
    problems = []
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties is not False")
        if sorted(schema.get("required", [])) != sorted(schema.get("properties", {}).keys()):
            problems.append(f"{path}: required does not list every property")
        for name, sub_schema in schema.get("properties", {}).items():
            problems.extend(find_strict_schema_violations(sub_schema, f"{path}.{name}"))
    elif schema.get("type") == "array":
        problems.extend(find_strict_schema_violations(schema["items"], f"{path}[]"))
    return problems


def assert_schema_is_strict_valid(schema):
    problems = find_strict_schema_violations(schema)
    assert not problems, "; ".join(problems)
