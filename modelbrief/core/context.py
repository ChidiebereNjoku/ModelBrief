from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DataSplit:
    X: Any = None
    y: Any = None

@dataclass
class ReportContext:
    model: Any
    train: DataSplit = field(default_factory=DataSplit)
    validation: DataSplit = field(default_factory=DataSplit)
    test: DataSplit = field(default_factory=DataSplit)
    task: str | None = None
    feature_names: list[str] | None = None
    target_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
