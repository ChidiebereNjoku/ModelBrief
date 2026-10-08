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
    missing values don't distort the whole calculation.

    Low-cardinality numeric columns (binary 0/1 flags, one-hot encoded
    categories, small integer codes -- anything with at most ``bins``
    distinct values) are binned by their exact value rather than by
    percentile. Percentile binning breaks down on these: a binary column
    only has 2 possible cut points no matter how much data you have, which
    would otherwise make it indistinguishable from a genuinely constant
    feature. Binning by exact value instead gives a real proportion-based
    PSI, so ordinary sampling noise (e.g. 65% vs. 61% in two random splits
    of the same population) correctly comes out small, while an actual
    category shift (e.g. a value that disappears entirely, or a new one
    that appears) is still caught.

    A genuinely constant feature (exactly 1 unique value in ``expected``)
    is a special case of the above: PSI is computed the same way, so a
    constant that shifts to a different constant value in ``actual`` is
    correctly flagged rather than silently reported as stable.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    unique_expected = np.unique(expected)

    if len(unique_expected) <= bins:
        # Discrete/categorical-like: bin by exact value, not by percentile.
        all_values = np.union1d(unique_expected, np.unique(actual))
        expected_counts = np.array([(expected == v).sum() for v in all_values])
        actual_counts = np.array([(actual == v).sum() for v in all_values])
        expected_pct = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
        actual_pct = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    cut_points = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(cut_points) < 3:
        # Continuous-looking but still too little spread to bin meaningfully.
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


def _image_channel_summaries(X):
    """Reduce an image-like array to one row of per-channel summary stats per image.

    Accepts channels-last arrays: (N, H, W) grayscale or (N, H, W, C). Returns
    (summary_array, names) where summary_array has shape (N, 2*C) -- mean and
    std per channel, per image -- or (None, None) if the array has more
    dimensions than that (e.g. video, or an unexpected layout) and isn't
    something this function knows how to summarize.

    Why mean/std rather than raw pixels: comparing every pixel position as
    its own "feature" (the naive approach) breaks down for images -- it
    silently treats a spatial axis as if it were tabular columns, producing
    a large number of meaningless, noisy comparisons. Per-image brightness
    (mean) and contrast (std) per channel are coarse but genuinely
    meaningful summaries: a camera/sensor change, a lighting shift, or a
    preprocessing bug reliably shows up in at least one of them. The
    trade-off is real and worth stating plainly: a purely localized change
    (e.g. only one corner of the image changes, with the rest compensating)
    can still be invisible to whole-image summary statistics. This is a
    deliberate, documented limitation, not a silent gap.
    """
    if X.ndim == 3:
        X = X[..., np.newaxis]
    if X.ndim != 4:
        return None, None

    n, h, w, c = X.shape
    columns = []
    names = []
    for ch in range(c):
        channel_pixels = X[:, :, :, ch].reshape(n, -1)
        columns.append(channel_pixels.mean(axis=1))
        names.append(f"channel_{ch}_mean")
        columns.append(channel_pixels.std(axis=1))
        names.append(f"channel_{ch}_std")

    return np.column_stack(columns), names


def dataset_drift(X_expected, X_actual, feature_names=None, bins=10):
    """Compare feature distributions between two dataset splits.

    Parameters
    ----------
    X_expected : array-like of shape (n_samples, n_features), or an
        image-like array of shape (n_samples, height, width[, channels])
        The reference split, typically training data.
    X_actual : array-like, same shape convention as X_expected
        The comparison split, typically validation or test data.
    feature_names : list[str], optional
        Names for each column; defaults to "feature_0", "feature_1", ...
        Must match the number of columns in ``X_expected`` if provided.
        Ignored for image-like input (see below), since per-pixel names
        don't apply to the derived per-channel summaries.
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
        has zero rows, ``feature_names`` doesn't match the column count,
        or the input has more dimensions than this function knows how to
        summarize (e.g. video).

        For image-like input (3D or 4D arrays), ``"image_summary": True``
        is included, "features" describes per-channel brightness (mean)
        and contrast (std) rather than raw pixels, and a ``"note"`` field
        explains the trade-off made.
    """
    try:
        X_expected = np.asarray(X_expected, dtype=float)
        X_actual = np.asarray(X_actual, dtype=float)
    except (ValueError, TypeError):
        return {"available": False, "reason": "non-numeric features are not yet supported"}

    image_summary_used = False

    if X_expected.ndim >= 3:
        if X_expected.ndim != X_actual.ndim:
            return {"available": False, "reason": "splits have a different number of dimensions"}

        summary_expected, derived_names = _image_channel_summaries(X_expected)
        summary_actual, _ = _image_channel_summaries(X_actual)

        if summary_expected is None:
            return {"available": False,
                    "reason": f"arrays with {X_expected.ndim} dimensions are not supported "
                              "(expected (N, H, W) or (N, H, W, C) image data)"}
        if summary_expected.shape[1] != summary_actual.shape[1]:
            return {"available": False, "reason": "splits have a different number of channels"}

        X_expected, X_actual = summary_expected, summary_actual
        feature_names = derived_names  # per-pixel names (if any were given) don't apply here
        image_summary_used = True

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

    result = {
        "available": True,
        "features": features,
        "significant_shift_count": sum(1 for f in features if f["status"] == "significant_shift"),
    }

    if image_summary_used:
        result["image_summary"] = True
        result["note"] = ("Input looked like image data, so drift was computed on per-channel "
                           "brightness (mean) and contrast (std) per image, not raw pixels. "
                           "A localized change affecting only part of the image may not show up here.")

    return result
