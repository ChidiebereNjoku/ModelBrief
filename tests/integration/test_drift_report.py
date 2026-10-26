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


def _report(n_train=300, n_test=150, drift_col=None, drift_shift=4.0, with_test=True):
    X_train, y_train = _make_split(n_train, 4, seed=1)
    model = RandomForestClassifier(n_estimators=20, random_state=1).fit(X_train, y_train)
    if not with_test:
        return ModelBrief(model, X_train=X_train, y_train=y_train)
    X_test, y_test = _make_split(n_test, 4, drift_col=drift_col, drift_shift=drift_shift, seed=2)
    return ModelBrief(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def test_drift_is_not_part_of_the_main_report():
    # analyse() / show() must never include drift -- it is a separate, opt-in method.
    report = _report(drift_col=2, drift_shift=5.0)
    result = report.analyse()

    assert result.section("DATA DRIFT") is None
    assert not any("drift" in f["title"].lower() for f in result.figures)
    assert "DATA DRIFT" not in report.show()


def test_data_drift_detects_a_real_shift():
    report = _report(n_train=800, n_test=300, drift_col=2, drift_shift=5.0)

    drift = report.data_drift()

    assert drift["available"] is True
    assert drift["significant_shift_count"] >= 1
    assert drift["features"][0]["feature"] == "feature_2"


def test_data_drift_unavailable_without_a_second_split():
    drift = _report(with_test=False).data_drift()

    assert drift["available"] is False


def test_data_drift_works_without_calling_analyse_first():
    report = _report(drift_col=1, drift_shift=4.0)

    drift = report.data_drift()

    assert drift["available"] is True
    assert report._result is None  # confirms analyse() was never triggered


def test_data_drift_plot_true_returns_a_figure():
    drift = _report(n_train=500, n_test=200, drift_col=0, drift_shift=4.0).data_drift(plot=True)

    assert drift["available"] is True
    assert isinstance(drift["figure"], str) and len(drift["figure"]) > 100


def test_data_drift_has_no_figure_by_default():
    assert "figure" not in _report().data_drift()


def test_data_drift_plot_does_not_need_a_gui_backend(monkeypatch):
    # Regression test: plot=True used to call pyplot, which opens a Tk/Qt
    # window and crashes on machines with no working GUI toolkit (seen on a
    # Windows Python with a broken Tcl install). Simulate that failure by
    # making pyplot's figure creation blow up -- the drift chart must not
    # depend on it.
    import matplotlib.pyplot as plt

    def _broken(*args, **kwargs):
        raise RuntimeError("no usable GUI backend")

    monkeypatch.setattr(plt, "subplots", _broken)
    monkeypatch.setattr(plt, "figure", _broken)

    drift = _report(n_train=500, n_test=200, drift_col=0, drift_shift=4.0).data_drift(plot=True)

    assert drift["available"] is True
    assert len(drift["figure"]) > 100
