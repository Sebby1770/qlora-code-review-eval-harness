"""Dataset IO helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, TypeVar

from review_tuner.schema import DatasetError, Prediction, ReviewExample


class HasId(Protocol):
    @property
    def id(self) -> str:
        """Stable record identifier."""


T = TypeVar("T", bound=HasId)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSON objects from a JSONL file without buffering the dataset."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"{source}:{row_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise DatasetError(f"{source}:{row_number}: expected a JSON object")
                yield value
    except DatasetError:
        raise
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"could not read dataset {source}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries.

    Prefer :func:`iter_jsonl` for production paths that do not require random access.
    """

    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically stream dictionaries to JSONL."""

    destination = Path(path)
    lines = (
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _atomic_write(destination, lines)


def write_json(path: str | Path, value: object) -> None:
    """Atomically write one pretty-printed JSON document."""

    destination = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_write(destination, [text])


def _atomic_write(destination: Path, chunks: Iterable[str]) -> None:
    """Write text chunks to a sibling temporary file and atomically replace the target."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if isinstance(exc, DatasetError):
            raise
        if not isinstance(exc, (OSError, TypeError, ValueError)):
            raise
        raise DatasetError(f"could not write {destination}: {exc}") from exc


def iter_examples(path: str | Path) -> Iterator[ReviewExample]:
    """Yield validated training or golden examples."""

    source = Path(path)
    for row_number, record in enumerate(iter_jsonl(source), start=1):
        try:
            yield ReviewExample.from_dict(record, row_number=row_number)
        except DatasetError as exc:
            raise DatasetError(f"{source}:{exc}") from exc


def load_examples(path: str | Path) -> list[ReviewExample]:
    """Load training or golden examples into memory."""

    return list(iter_examples(path))


def iter_predictions(path: str | Path) -> Iterator[Prediction]:
    """Yield validated model predictions."""

    source = Path(path)
    for row_number, record in enumerate(iter_jsonl(source), start=1):
        try:
            yield Prediction.from_dict(record, row_number=row_number)
        except DatasetError as exc:
            raise DatasetError(f"{source}:{exc}") from exc


def load_predictions(path: str | Path) -> list[Prediction]:
    """Load model predictions into memory."""

    return list(iter_predictions(path))


def index_by_id(items: Iterable[T]) -> dict[str, T]:
    """Index dataclass-like objects by their id attribute."""

    indexed: dict[str, T] = {}
    for item in items:
        item_id = item.id
        if item_id in indexed:
            raise DatasetError(f"duplicate id: {item_id}")
        indexed[item_id] = item
    return indexed
