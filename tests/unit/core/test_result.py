from modelbrief.core.context import DataSplit, ReportContext
from modelbrief.core.result import ReportResult, Section


def test_add_returns_result_so_calls_can_be_chained():
    result = ReportResult()
    returned = result.add("A", {"x": 1}).add("B", {"y": 2}, narrative="note")

    assert returned is result
    assert [s.title for s in result.sections] == ["A", "B"]
    assert result.sections[1].narrative == "note"


def test_section_finds_by_title_or_returns_none():
    result = ReportResult().add("MODEL PERFORMANCE", {"accuracy": 1.0})

    assert result.section("MODEL PERFORMANCE").content == {"accuracy": 1.0}
    assert result.section("MISSING") is None


def test_to_dict_has_every_part_of_the_report():
    result = ReportResult(metadata={"task": "classification"})
    result.add("TASK", {"detected_or_requested": "classification"})
    result.warnings.append("a warning")
    result.recommendations.append("a recommendation")
    result.figures.append({"title": "chart", "data": "abc"})

    assert result.to_dict() == {
        "metadata": {"task": "classification"},
        "sections": [
            {
                "title": "TASK",
                "content": {"detected_or_requested": "classification"},
                "narrative": None,
            }
        ],
        "figures": [{"title": "chart", "data": "abc"}],
        "warnings": ["a warning"],
        "recommendations": ["a recommendation"],
    }


def test_results_do_not_share_lists():
    first = ReportResult()
    second = ReportResult()
    first.warnings.append("only on first")

    assert second.warnings == []


def test_section_defaults():
    section = Section("TITLE")

    assert section.content == {}
    assert section.narrative is None


def test_context_defaults():
    context = ReportContext(model=object())

    for split in (context.train, context.validation, context.test):
        assert isinstance(split, DataSplit)
        assert split.X is None and split.y is None

    assert context.task is None
    assert context.options == {}
    assert ReportContext(model=object()).train is not context.train
