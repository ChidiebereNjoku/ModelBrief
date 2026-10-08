import numpy as np
from sklearn.ensemble import RandomForestClassifier

from modelbrief import ModelBrief


def _make_split(n, n_features, drift_col=None, drift_shift=4.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    if drift_col is not None:
        X[:, drift_col] += drift_shift
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_drift_detected_between_train_and_test():
    X_train, y_train = _make_split(800, 4, seed=1)
    X_test, y_test = _make_split(300, 4, drift_col=2, drift_shift=5.0, seed=2)

    model = RandomForestClassifier(n_estimators=50, random_state=1).fit(X_train, y_train)
    result = ModelBrief(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test).analyse()

    drift = result.section("DATA DRIFT").content
    assert drift["available"] is True
    assert drift["significant_shift_count"] >= 1
    assert drift["features"][0]["feature"] == "feature_2"
    assert any(f["title"] == "Data drift" for f in result.figures)


def test_drift_unavailable_without_a_second_split():
    X_train, y_train = _make_split(300, 4, seed=1)

    model = RandomForestClassifier(n_estimators=20, random_state=1).fit(X_train, y_train)
    result = ModelBrief(model, X_train=X_train, y_train=y_train).analyse()

    drift = result.section("DATA DRIFT").content
    assert drift["available"] is False
    assert not any(f["title"] == "Data drift" for f in result.figures)
