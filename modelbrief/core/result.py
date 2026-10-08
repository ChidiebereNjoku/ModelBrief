from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Section:
    title: str
    content: dict[str, Any] = field(default_factory=dict)
    narrative: str | None = None

@dataclass
class ReportResult:
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    def add(self, title: str, content: dict[str, Any], narrative: str | None = None):
        self.sections.append(Section(title, content, narrative)); return self
    def section(self, title: str):
        return next((s for s in self.sections if s.title == title), None)
    def to_dict(self):
        return {"metadata":self.metadata,"sections":[{"title":s.title,"content":s.content,"narrative":s.narrative} for s in self.sections],"figures":self.figures,"warnings":self.warnings,"recommendations":self.recommendations}
