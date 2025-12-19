from __future__ import annotations

import json
import re
from typing import Any, Dict


def safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse a JSON object from an LLM response.

    Behavior:
    - Strips common ``` / ```json code fences
    - If extra text surrounds JSON, extracts the outermost {...} block
    """

    s = (text or "").strip()

    # Strip code fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)

    # Extract first JSON object
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first : last + 1]

    return json.loads(s)
