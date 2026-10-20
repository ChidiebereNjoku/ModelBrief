import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from modelbrief import ModelBrief


class FixedClassifier:
    """Fake classifier that always returns the same predictions."""

    _estimator_type = "classifier"

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions[: len(X)]


class FixedRegressor:
    _estimator_type = "regressor"

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions[: len(X)]


class BrokenModel:
    """Fake classifier whose predict always fails."""

    _estimator_type = "classifier"

    def predict(self, X):
        raise ValueError("bad input")


class ImportanceModel:
    _estimator_type = "classifier"
    feature_importances_ = (0.1, 0.7, 0.2)

    def predict(self, X):
        return [0, 1][: len(X)]


class FrameLike:
    """Small stand-in for a pandas DataFrame, so pandas is not needed."""

    def __init__(self, data, columns):
        self.data = np.asarray(data)
        self.columns = columns
        self.shape = self.data.shape

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


def test_analyse_result_is_cached_until_forced():
    report = ModelBrief(FixedClassifier([0, 1]), X_test=[[1], [2]], y_test=[0, 1])
    first = report.analyse()

    assert report.analyse() is first
    assert report.result is first
    assert report.analyse(force=True) is not first


def test_metadata_has_task_and_version():
    result = ModelBrief(FixedClassifier([0, 1]), X_test=[[1], [2]], y_test=[0, 1]).analyse()

    assert result.metadata["task"] == "classification"
    assert result.metadata["library_version"] == "0.2.2"


def test_requested_task_overrides_detection():
    result = ModelBrief(
        FixedClassifier([1.0, 2.0]), X_test=[[1], [2]], y_test=[1.0, 2.0], task="regression"
    ).analyse()

    assert result.metadata["task"] == "regression"
    assert "rmse" in result.section("MODEL PERFORMANCE").content["test"]


def test_failed_split_becomes_a_warning_not_a_crash():
    result = ModelBrief(BrokenModel(), X_test=[[1], [2]], y_test=[0, 1]).analyse()

    assert "test analysis skipped: ValueError: bad input" in result.warnings
    assert result.section("MODEL PERFORMANCE").content == {}
    assert result.section("ERROR ANALYSIS").content == {"available": False, "reason": "bad input"}


def test_every_supplied_split_is_reported():
    X = [[1], [2]]
    y = [0, 1]
    result = ModelBrief(
        FixedClassifier([0, 1]), X_train=X, y_train=y, X_val=X, y_val=y, X_test=X, y_test=y
    ).analyse()

    assert set(result.section("MODEL PERFORMANCE").content) == {"train", "validation", "test"}


def test_split_without_labels_only_records_predictions():
    result = ModelBrief(FixedClassifier([0, 1]), X_train=[[1], [2]], y_train=[0, 1], X_test=[[1], [2]]).analyse()

    assert result.section("MODEL PERFORMANCE").content["test"] == {
        "samples": 2,
        "predictions_generated": True,
    }


def test_classification_error_analysis_lists_wrong_predictions():
    result = ModelBrief(FixedClassifier([0, 1, 0, 0]), X_test=[[1], [2], [3], [4]], y_test=[0, 1, 1, 0]).analyse()
    errors = result.section("ERROR ANALYSIS").content

    assert errors["available"] is True
    assert errors["error_count"] == 1
    assert errors["examples"] == [{"index": 2, "actual": "1", "predicted": "0"}]


def test_regression_error_analysis_puts_largest_error_first():
    y = [float(i) for i in range(30)]
    predictions = list(y)
    predictions[7] += 5.0
    predictions[3] += 1.0
    result = ModelBrief(FixedRegressor(predictions), X_test=[[i] for i in range(30)], y_test=y).analyse()
    largest = result.section("ERROR ANALYSIS").content["largest"]

    assert largest[0]["index"] == 7
    assert largest[0]["absolute_error"] == 5.0
    assert largest[1]["index"] == 3


def test_feature_names_are_taken_from_dataframe_columns():
    X = FrameLike([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    report = ModelBrief(ImportanceModel(), X_test=X, y_test=[0, 1])
    ranking = report.analyse().section("FEATURE IMPORTANCE").content["ranking"]

    assert report.context.feature_names == ["a", "b", "c"]
    assert [name for name, _ in ranking] == ["b", "c", "a"]


def test_explicit_feature_names_are_not_replaced():
    X = FrameLike([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    report = ModelBrief(ImportanceModel(), X_test=X, y_test=[0, 1], feature_names=["x", "y", "z"])

    assert report.context.feature_names == ["x", "y", "z"]


def test_classifier_report_has_confusion_matrix_and_figures():
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    model = LogisticRegression().fit(X[:60], y[:60])
    result = ModelBrief(model, X_test=X[60:], y_test=y[60:]).analyse()
    titles = [figure["title"] for figure in result.figures]

    assert result.section("CONFUSION MATRIX").content["available"] is True
    assert "Feature importance" in titles
    assert "Confusion matrix" in titles
    assert all(figure["data"] for figure in result.figures)


def test_regression_report_has_residuals_figure():
    X, y = make_regression(n_samples=60, n_features=3, random_state=0)
    model = LinearRegression().fit(X[:40], y[:40])
    result = ModelBrief(model, X_test=X[40:], y_test=y[40:]).analyse()

    assert "Residuals" in [figure["title"] for figure in result.figures]
    assert result.section("CONFUSION MATRIX").content["available"] is False


def test_clustering_without_labels_gets_cluster_metrics():
    X, _ = make_blobs(n_samples=40, centers=2, random_state=0)
    model = KMeans(n_clusters=2, n_init=1, random_state=0).fit(X)
    performance = ModelBrief(model, X_test=X).analyse().section("MODEL PERFORMANCE").content["test"]

    assert performance["clusters"] == 2
    assert "silhouette" in performance


def test_ai_is_off_by_default():
    result = ModelBrief(FixedClassifier([0, 1]), X_test=[[1], [2]], y_test=[0, 1]).analyse()

    assert result.section("AI EXPLANATION") is None
    assert result.recommendations == []
