"""Optional on-disk store for MCP `tools/list` pins (trust on first use across connections).

Without a pin file the proxies pin `tools/list` definitions per process, so a server that changes
its tool descriptions between two connections goes unnoticed. With a pin file the first-seen
SHA-256 of each tool definition is kept, and a later connection that sees a different digest takes
the same rug-pull path as a mid-session mutation. The file holds digests only, never tool text.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Final

_FORMAT: Final[str] = "epistemic-guard-tool-pins/1"
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def load_tool_pins(path: Path | str, endpoint_id: str) -> dict[str, str]:
    """Return the pins saved in `path` for `endpoint_id`; an absent file means no pins yet.

    Fails closed: an unreadable, malformed or other-endpoint file raises `ValueError` instead of
    silently starting from empty pins, because an empty start would re-trust a changed server.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"pin file {p} is unreadable or not JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != _FORMAT:
        raise ValueError(f"pin file {p} is not an {_FORMAT} file.")
    if data.get("endpoint_id") != endpoint_id:
        raise ValueError(
            f"pin file {p} was written for endpoint {data.get('endpoint_id')!r}, "
            f"not {endpoint_id!r}; use one pin file per server."
        )
    pins = data.get("pins")
    if not isinstance(pins, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and _SHA256_RE.match(v) for k, v in pins.items()
    ):
        raise ValueError(f"pin file {p} has a malformed `pins` table.")
    return dict(pins)


def save_tool_pins(path: Path | str, endpoint_id: str, pins: dict[str, str]) -> None:
    """Atomically write `pins` for `endpoint_id` to `path` (temp file, then replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    payload = {"format": _FORMAT, "endpoint_id": endpoint_id, "pins": dict(sorted(pins.items()))}
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
