import json
import math
import os
import re
from pathlib import Path

KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

def toml_key(key):
    if KEY_RE.match(key):
        return key
    return json.dumps(key)

def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"TOML does not support non-finite float: {value}")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")

def emit_table(lines, path, table):
    scalar_items = []
    child_items = []
    for key, value in table.items():
        if isinstance(value, dict):
            child_items.append((key, value))
        elif value is None:
            raise TypeError(f"TOML does not support null: {'.'.join(path + [key])}")
        else:
            scalar_items.append((key, value))

    if path:
        lines.append(f"[{'.'.join(toml_key(part) for part in path)}]")
    for key, value in scalar_items:
        lines.append(f"{toml_key(key)} = {toml_value(value)}")

    if scalar_items and child_items:
        lines.append("")
    for index, (key, value) in enumerate(child_items):
        emit_table(lines, path + [key], value)
        if index != len(child_items) - 1:
            lines.append("")

config = json.loads(os.environ["HYPERPARAMETERS"])
if not isinstance(config, dict):
    raise TypeError("HYPERPARAMETERS must be a JSON object")

lines = []
emit_table(lines, [], config)

output_path = Path(os.environ["TORCHTITAN_CONFIG_PATH"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(f"Wrote TorchTitan config TOML to {output_path}")