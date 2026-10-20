import numpy as np

from modelbrief.adapters.base import BaseAdapter
from modelbrief.adapters.tabular.sklearn import SklearnAdapter


class Adapter(BaseAdapter):
    def predict(self, X):
        return np.asarray(self.model.predict(X))


class WithParams:
    def get_params(self, deep=True):
        return {"alpha": 0.5, "deep": deep}


class BrokenParamsWithCount:
    def get_params(self, deep=True):
        raise RuntimeError("cannot read params")

    def count_params(self):
        return 42


class Bare:
    pass


class WithImportances:
    feature_importances_ = np.array([0.2, 0.5, 0.3])


class WithCoef1D:
    coef_ = np.array([0.5, -2.0, 1.0])


class WithCoef2D:
    coef_ = np.array([[1.0, -3.0], [-1.0, 1.0]])


class WithProba:
    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return [[0.9, 0.1]] * len(X)


def test_parameters_use_get_params_without_deep():
    assert Adapter(WithParams()).parameters() == {"alpha": 0.5, "deep": False}


def test_parameters_fall_back_to_count_params():
    assert Adapter(BrokenParamsWithCount()).parameters() == {"trainable_or_total_parameters": 42}


def test_parameters_fall_back_to_class_name():
    assert Adapter(Bare()).parameters() == {"class": "Bare"}


def test_feature_importance_is_sorted_with_default_names():
    assert Adapter(WithImportances()).feature_importance() == [
        ("feature_1", 0.5),
        ("feature_2", 0.3),
        ("feature_0", 0.2),
    ]


def test_feature_importance_uses_given_names():
    ranking = Adapter(WithImportances()).feature_importance(["a", "b", "c"])

    assert [name for name, _ in ranking] == ["b", "c", "a"]


def test_feature_importance_from_1d_coef_uses_absolute_values():
    assert Adapter(WithCoef1D()).feature_importance() == [
        ("feature_1", 2.0),
        ("feature_2", 1.0),
        ("feature_0", 0.5),
    ]


def test_feature_importance_from_2d_coef_averages_absolute_values():
    assert Adapter(WithCoef2D()).feature_importance() == [
        ("feature_1", 2.0),
        ("feature_0", 1.0),
    ]


def test_feature_importance_is_none_when_model_has_none():
    assert Adapter(Bare()).feature_importance() is None


def test_predict_proba_is_none_when_model_has_no_method():
    assert Adapter(Bare()).predict_proba([[1]]) is None
    assert SklearnAdapter(Bare()).predict_proba([[1]]) is None


def test_predict_proba_returns_an_array():
    proba = SklearnAdapter(WithProba()).predict_proba([[1], [2]])

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (2, 2)


def test_sklearn_adapter_predict_returns_an_array():
    predictions = SklearnAdapter(WithProba()).predict([[1], [2], [3]])

    assert isinstance(predictions, np.ndarray)
    assert predictions.tolist() == [0, 0, 0]
