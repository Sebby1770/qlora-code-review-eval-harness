"""Dataset IO helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, TypeVar

from review_tuner.schema import DatasetError, Prediction, ReviewExample


class HasId(Protocol):
    id: str


T = TypeVar("T", bound=HasId)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""

    records: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{row_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise DatasetError(f"{path}:{row_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_examples(path: str | Path) -> list[ReviewExample]:
    """Load training or golden examples."""

    return [
        ReviewExample.from_dict(record, row_number=index)
        for index, record in enumerate(read_jsonl(path), start=1)
    ]


def load_predictions(path: str | Path) -> list[Prediction]:
    """Load model predictions."""

    return [
        Prediction.from_dict(record, row_number=index)
        for index, record in enumerate(read_jsonl(path), start=1)
    ]


def index_by_id(items: Iterable[T]) -> dict[str, T]:
    """Index dataclass-like objects by their id attribute."""

    indexed: dict[str, T] = {}
    for item in items:
        item_id = item.id
        if item_id in indexed:
            raise DatasetError(f"duplicate id: {item_id}")
        indexed[item_id] = item
    return indexed
