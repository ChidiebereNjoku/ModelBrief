from modelbrief import ModelBrief
from modelbrief.output.console import ConsoleReport


class DummyEstimator:
    _estimator_type = "classifier"

    def predict(self, X):
        return [0, 1][:len(X)]


class DummyPrettyPrinter:
    def __init__(self):
        self.output = ""

    def text(self, val):
        self.output += val


def test_show_does_not_print_internally(capsys):
    report = ModelBrief(model=DummyEstimator(), X_test=[[1], [2]], y_test=[0, 1])
    result = report.show()
    captured = capsys.readouterr()

    assert captured.out == ""
    assert isinstance(result, str)
    assert "MODELBRIEF REPORT" in result
    assert "MODEL OVERVIEW" in result


def test_print_show_outputs_report_exactly_once(capsys):
    report = ModelBrief(model=DummyEstimator(), X_test=[[1], [2]], y_test=[0, 1])
    print(report.show())
    captured = capsys.readouterr()

    assert captured.out.count("MODELBRIEF REPORT") == 1
    assert captured.out.count("MODEL OVERVIEW") == 1


def test_console_report_pretty_repr():
    report_text = "MODELBRIEF REPORT\nTEST CONTENT"
    console_report = ConsoleReport(report_text)
    printer = DummyPrettyPrinter()
    console_report._repr_pretty_(printer, cycle=False)

    assert printer.output == report_text
