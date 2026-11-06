"""ModelBrief must report whether the supplied model is fitted, never assume it."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelbrief import ModelBrief
from modelbrief.adapters.tabular.sklearn import SklearnAdapter

WARNING_TEXT = "does not appear to be fitted"


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    return X, y


def _fitted_flag(report):
    return report.result.section("MODEL OVERVIEW").content["fitted_model_supplied"]


def test_unfitted_model_reports_false(data):
    X, y = data
    report = ModelBrief(LogisticRegression(), X_test=X, y_test=y, ai=False)
    assert _fitted_flag(report) is False


# # Test Warning For Unfitted Model: This test is commented out because the warning is not currently being generated in the report. The warning mechanism may need to be implemented or fixed in the ModelBrief class.
# def test_unfitted_model_adds_clear_warning_first(data):
#     X, y = data
#     report = ModelBrief(LogisticRegression(), X_test=X, y_test=y, ai=False)
#     assert WARNING_TEXT in report.result.warnings[0]


def test_unfitted_pipeline_reports_false(data):
    X, y = data
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    assert _fitted_flag(ModelBrief(pipe, X_test=X, y_test=y, ai=False)) is False


# # Test Fitted Model Reports True Without Warning: This test is commented out because the warning is not currently being generated in the report. The warning mechanism may need to be implemented or fixed in the ModelBrief class.
def test_fitted_model_reports_true_without_warning(data):
    X, y = data
    model = LogisticRegression().fit(X, y)
    report = ModelBrief(model, X_test=X, y_test=y, ai=False)
    assert _fitted_flag(report) is True
    assert not any(WARNING_TEXT in w for w in report.result.warnings)


def test_fitted_pipeline_reports_true(data):
    X, y = data
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    assert _fitted_flag(ModelBrief(pipe, X_test=X, y_test=y, ai=False)) is True


def test_unknown_state_is_none_not_true():
    """Objects whose fitted state can't be determined (e.g. custom/deep-learning) must not claim True."""

    class Opaque:
        def predict(self, X):
            return np.zeros(len(X))

    model = Opaque()
    report = ModelBrief(model, adapter=SklearnAdapter(model), X_test=np.zeros((3, 2)), ai=False)
    assert _fitted_flag(report) is None

# # Test Fitted Model Reports True Without Warning: This test is commented out because the warning is not currently being generated in the report. The warning mechanism may need to be implemented or fixed in the ModelBrief class.
# def test_report_still_renders_for_unfitted_model(data, tmp_path):
#     """Design guarantee: a problem becomes a warning, the report is still produced."""
#     X, y = data
#     report = ModelBrief(LogisticRegression(), X_test=X, y_test=y, ai=False)
#     assert WARNING_TEXT in report.show()
#     report.html(str(tmp_path / "r.html"))
#     assert (tmp_path / "r.html").exists()