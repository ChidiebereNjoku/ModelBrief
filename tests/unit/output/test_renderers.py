import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelbrief.core.result import ReportResult
from modelbrief.output.console import ConsoleReport, render_console
from modelbrief.output.html import render_html
from modelbrief.output.pdf import render_pdf


def tiny_png():
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def sample_result(**extra):
    result = ReportResult(metadata={"task": "classification"})
    result.add("MODEL OVERVIEW", {"model_class": "Dummy"})
    result.add("MODEL PERFORMANCE", {"test": {"accuracy": 0.75}}, narrative="Looks fine.")
    for key, value in extra.items():
        setattr(result, key, value)
    return result


def test_console_lists_sections_and_narrative():
    text = render_console(sample_result())

    assert isinstance(text, ConsoleReport)
    assert "MODEL OVERVIEW" in text
    assert "MODEL PERFORMANCE" in text
    assert '"accuracy": 0.75' in text
    assert "Looks fine." in text


def test_console_shows_warnings_and_recommendations_only_when_present():
    plain = render_console(sample_result())
    full = render_console(sample_result(warnings=["test analysis skipped"], recommendations=["Get more data"]))

    assert "WARNINGS" not in plain
    assert "RECOMMENDATIONS" not in plain
    assert "- test analysis skipped" in full
    assert "- Get more data" in full


def test_html_is_written_and_creates_folders(tmp_path):
    path = render_html(sample_result(), tmp_path / "nested" / "report.html")
    page = path.read_text(encoding="utf-8")

    assert path.exists()
    assert page.lstrip().startswith("<!doctype html>")
    assert "<h2>MODEL OVERVIEW</h2>" in page
    assert "<h2>MODEL PERFORMANCE</h2>" in page
    assert "<p>Looks fine.</p>" in page


def test_html_escapes_report_content(tmp_path):
    result = ReportResult()
    result.add("<b>TITLE</b>", {"<key>": "<script>alert(1)</script>"})
    page = render_html(result, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;b&gt;TITLE&lt;/b&gt;" in page


def test_html_includes_recommendations_and_figures(tmp_path):
    result = sample_result(
        recommendations=["Get more data"],
        figures=[{"title": "Confusion matrix", "data": tiny_png()}],
    )
    page = render_html(result, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "<h2>RECOMMENDATIONS</h2>" in page
    assert "<li>Get more data</li>" in page
    assert 'src="data:image/png;base64,' in page
    assert "<figcaption>Confusion matrix</figcaption>" in page


def test_pdf_is_written_and_creates_folders(tmp_path):
    path = render_pdf(sample_result(), tmp_path / "nested" / "report.pdf")

    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_pdf_with_figures_and_recommendations(tmp_path):
    plain = render_pdf(sample_result(), tmp_path / "plain.pdf")
    full = render_pdf(
        sample_result(
            recommendations=["Get more data"],
            figures=[{"title": "Residuals", "data": tiny_png()}],
        ),
        tmp_path / "full.pdf",
    )

    assert full.read_bytes().startswith(b"%PDF")
    assert full.stat().st_size > plain.stat().st_size


def test_pdf_skips_a_broken_figure_instead_of_failing(tmp_path):
    result = sample_result(figures=[{"title": "Broken"}])
    path = render_pdf(result, tmp_path / "report.pdf")

    assert path.read_bytes().startswith(b"%PDF")
