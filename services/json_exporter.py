from __future__ import annotations

import json
from typing import Any

from core.document import to_json
from core.types import VectorDocument


class JsonExporter:
    def __init__(self, *, indent: int = 2, sort_keys: bool = True) -> None:
        self.indent = indent
        self.sort_keys = sort_keys

    def export_document(self, document: VectorDocument) -> str:
        payload = json.loads(to_json(document))
        return json.dumps(payload, indent=self.indent, sort_keys=self.sort_keys)

    def export_to_dict(self, document: VectorDocument) -> dict[str, Any]:
        return json.loads(to_json(document))


__all__ = ["JsonExporter"]
