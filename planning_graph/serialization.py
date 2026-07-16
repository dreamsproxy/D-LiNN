"""JSON persistence helpers for planning graph documents."""

from __future__ import annotations

import json
from pathlib import Path

from .models import PlanningDocument


def save_document(document: PlanningDocument, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.touch()
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(document.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return destination


def load_document(path: str | Path) -> PlanningDocument:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PlanningDocument.from_dict(payload)
