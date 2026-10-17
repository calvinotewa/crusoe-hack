"""Small .env helper used by scripts/setup.sh.

    dotenv_tool.py normalize ENV TEMPLATE      build ENV from TEMPLATE, keeping existing values
    dotenv_tool.py get ENV KEY                 print a value ("" if unset)
    dotenv_tool.py set ENV KEY  < value        value via stdin (keeps secrets out of argv)
    dotenv_tool.py merge ENV SRC KEY...        copy KEYs that are non-empty in SRC into ENV
    dotenv_tool.py modal-tokens ENV            copy the active Modal token into ENV

Writes are atomic and leave the file chmod 600.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def quote(value: str) -> str:
    # Single quotes: python-dotenv does no ${VAR} interpolation inside them (keys may contain $).
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def write(path: Path, lines: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".env.tmp.")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def set_values(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    pending = dict(values)
    for i, line in enumerate(lines):
        m = KEY_RE.match(line)
        if m and m.group(1) in pending:
            key = m.group(1)
            lines[i] = f"{key}={quote(pending.pop(key))}"
    lines += [f"{k}={quote(v)}" for k, v in pending.items()]
    write(path, lines)


def normalize(path: Path, template: Path) -> None:
    """Every template key present, in template order with its comments. Existing lines
    win over template defaults, and keys only in the existing file are kept at the end."""
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if m := KEY_RE.match(line):
                existing[m.group(1)] = line
    out, seen = [], set()
    for line in template.read_text().splitlines():
        m = KEY_RE.match(line)
        if m and m.group(1) in existing:
            out.append(existing[m.group(1)])
            seen.add(m.group(1))
        else:
            out.append(line)
    extra = [line for key, line in existing.items() if key not in seen]
    if extra:
        out += ["", "# --- Local additions ---", *extra]
    write(path, out)


def modal_tokens() -> dict[str, str]:
    from modal.config import config  # active profile in ~/.modal.toml, or MODAL_TOKEN_* env

    token_id, token_secret = config.get("token_id"), config.get("token_secret")
    if not (token_id and token_secret):
        sys.exit("no Modal token found; run `uv run modal token new`")
    return {"MODAL_TOKEN_ID": token_id, "MODAL_TOKEN_SECRET": token_secret}


def main() -> None:
    cmd, path, *rest = sys.argv[1:]
    env = Path(path)
    if cmd == "normalize":
        normalize(env, Path(rest[0]))
    elif cmd == "get":
        values = dotenv_values(env) if env.exists() else {}
        print(values.get(rest[0]) or "")
    elif cmd == "set":
        set_values(env, {rest[0]: sys.stdin.read()})
    elif cmd == "merge":
        src = dotenv_values(rest[0])
        set_values(env, {k: src[k] for k in rest[1:] if src.get(k)})
    elif cmd == "modal-tokens":
        set_values(env, modal_tokens())
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
