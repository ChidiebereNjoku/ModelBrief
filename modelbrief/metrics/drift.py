"""Distribution-drift detection between dataset splits.

Every other analysis in ModelBrief evaluates a model against whatever test
data it was given, implicitly trusting that the test set is a fair stand-in
for training data (and, by extension, for future/production data). This
module checks that assumption directly by comparing the feature
distributions of two splits (typically train vs. test) using the
Population Stability Index (PSI), a standard drift metric used in
production ML monitoring.

A model can score well on accuracy, calibration, and every other metric in
the report while still being evaluated on a test set that no longer
resembles the data it was trained on -- this module exists to catch that
case, which none of the other analyses can see.
"""

import numpy as np


def population_stability_index(expected, actual, bins=10):
    """Compute the Population Stability Index between two 1D numeric samples.

    PSI buckets the ``expected`` (reference/train) sample into ``bins``
    equal-frequency bins, then compares how the ``actual`` (comparison/test)
    sample falls into those same bins. Common interpretation thresholds:

    - PSI < 0.1  -> no significant shift
    - 0.1 <= PSI < 0.25 -> moderate shift, worth a look
    - PSI >= 0.25 -> significant shift

    NaN/inf values are dropped from each sample before binning, so a few
    missing values don't distort the whole calculation. If the reference
    sample has too little variation to bin meaningfully (e.g. a constant
    feature), PSI falls back to a simple "did the value change at all"
    check rather than silently reporting 0.0 for a feature that may have
    shifted entirely (e.g. a constant 1 in training vs. a constant 100 in
    the comparison sample).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    cut_points = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(cut_points) < 3:
        # Not enough variation in `expected` to bin meaningfully (e.g. a
        # constant or near-constant feature). Fall back to comparing means
        # directly so a real shift (e.g. constant-to-constant at a
        # different value) is still flagged rather than reported as stable.
        expected_value = float(np.mean(expected))
        actual_value = float(np.mean(actual))
        tolerance = max(1e-9, abs(expected_value) * 1e-6)
        return 0.0 if abs(actual_value - expected_value) <= tolerance else 1.0

    # Widen the outer edges slightly so min/max values of `actual` aren't dropped.
    cut_points[0] -= 1e-6
    cut_points[-1] += 1e-6

    expected_counts, _ = np.histogram(expected, bins=cut_points)
    actual_counts, _ = np.histogram(actual, bins=cut_points)

    expected_pct = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _status(psi):
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_shift"


def dataset_drift(X_expected, X_actual, feature_names=None, bins=10):
    """Compare feature distributions between two dataset splits.

    Parameters
    ----------
    X_expected : array-like of shape (n_samples, n_features)
        The reference split, typically training data.
    X_actual : array-like of shape (n_samples, n_features)
        The comparison split, typically validation or test data.
    feature_names : list[str], optional
        Names for each column; defaults to "feature_0", "feature_1", ...
        Must match the number of columns in ``X_expected`` if provided.
    bins : int, default=10
        Number of bins used for the PSI calculation.

    Returns
    -------
    dict
        ``{"available": True, "features": [...]}`` where each entry is
        ``{"feature": str, "psi": float, "status": str}``, sorted by PSI
        descending (most-drifted feature first). Returns
        ``{"available": False, "reason": str}`` instead of raising if:
        the dataset contains non-numeric values (the whole comparison is
        currently unsupported in that case, not skipped per-column),
        the two splits have a different number of features, either split
        has zero rows, or ``feature_names`` doesn't match the column count.
    """
    try:
        X_expected = np.asarray(X_expected, dtype=float)
        X_actual = np.asarray(X_actual, dtype=float)
    except (ValueError, TypeError):
        return {"available": False, "reason": "non-numeric features are not yet supported"}

    if X_expected.ndim == 1:
        X_expected = X_expected.reshape(-1, 1)
    if X_actual.ndim == 1:
        X_actual = X_actual.reshape(-1, 1)

    if X_expected.shape[0] == 0 or X_actual.shape[0] == 0:
        return {"available": False, "reason": "one of the splits has zero rows"}

    if X_expected.shape[1] != X_actual.shape[1]:
        return {"available": False, "reason": "splits have a different number of features"}

    if feature_names is not None and len(feature_names) != X_expected.shape[1]:
        return {"available": False, "reason": "feature_names length does not match the number of columns"}

    names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(X_expected.shape[1])]

    features = []
    for i, name in enumerate(names):
        psi = population_stability_index(X_expected[:, i], X_actual[:, i], bins=bins)
        features.append({"feature": name, "psi": psi, "status": _status(psi)})

    features.sort(key=lambda f: f["psi"], reverse=True)

    return {
        "available": True,
        "features": features,
        "significant_shift_count": sum(1 for f in features if f["status"] == "significant_shift"),
    }
