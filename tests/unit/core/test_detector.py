import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelbrief.adapters.tabular.sklearn import SklearnAdapter
from modelbrief.core.detector import detect_adapter, detect_task


class Typed:
    """Fake model that only exposes an estimator type."""

    def __init__(self, estimator_type):
        self._estimator_type = estimator_type


class Plain:
    """Fake model with no estimator type at all."""


class Tags:
    estimator_type = "regressor"


class TaggedModel:
    """Fake model that uses the newer __sklearn_tags__ API."""

    def __sklearn_tags__(self):
        return Tags()


class BrokenTags:
    """Fake model whose __sklearn_tags__ fails."""

    def __sklearn_tags__(self):
        raise TypeError("no tags")


class HasLabels:
    labels_ = (0, 1, 0)


def test_requested_task_wins_and_is_normalised():
    assert detect_task(Typed("classifier"), requested=" Time Series ") == "time_series"


def test_estimator_type_classifier():
    assert detect_task(Typed("classifier")) == "classification"


def test_estimator_type_regressor():
    assert detect_task(Typed("regressor")) == "regression"


def test_estimator_type_clusterer():
    assert detect_task(Typed("clusterer")) == "clustering"


def test_sklearn_tags_are_used_when_no_estimator_type():
    assert detect_task(TaggedModel()) == "regression"


def test_broken_sklearn_tags_fall_back_to_target():
    assert detect_task(BrokenTags(), y=[0, 1, 0, 1]) == "classification"


def test_labels_attribute_means_clustering():
    assert detect_task(HasLabels(), y=[1.5, 2.5, 3.5]) == "clustering"


def test_no_target_means_clustering():
    assert detect_task(Plain()) == "clustering"


def test_string_and_bool_targets_are_classification():
    assert detect_task(Plain(), y=["cat", "dog", "cat"]) == "classification"
    assert detect_task(Plain(), y=[True, False, True]) == "classification"


def test_few_unique_numbers_are_classification():
    assert detect_task(Plain(), y=list(range(10)) * 5) == "classification"


def test_many_unique_numbers_are_regression():
    assert detect_task(Plain(), y=np.linspace(0, 1, 100)) == "regression"


def test_real_sklearn_estimators():
    assert detect_task(LogisticRegression()) == "classification"
    assert detect_task(LinearRegression()) == "regression"
    assert detect_task(KMeans(n_clusters=2, n_init=1)) == "clustering"


def test_sklearn_models_and_pipelines_use_sklearn_adapter():
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    assert isinstance(detect_adapter(LogisticRegression()), SklearnAdapter)
    assert isinstance(detect_adapter(pipeline), SklearnAdapter)


def test_unknown_models_fall_back_to_sklearn_adapter():
    adapter = detect_adapter(Plain())
    assert isinstance(adapter, SklearnAdapter)
    assert isinstance(adapter.model, Plain)
