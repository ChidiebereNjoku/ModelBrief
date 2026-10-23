import numpy as np

from modelbrief.metrics.drift import population_stability_index, dataset_drift


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=2000)

    psi = population_stability_index(sample, sample)

    assert psi < 0.01


def test_psi_flags_a_clear_shift():
    rng = np.random.default_rng(1)
    expected = rng.normal(loc=0, scale=1, size=2000)
    actual = rng.normal(loc=3, scale=1, size=2000)  # shifted well away from `expected`

    psi = population_stability_index(expected, actual)

    assert psi >= 0.25


def test_dataset_drift_ranks_the_shifted_feature_first():
    rng = np.random.default_rng(1)
    n = 1500
    X_train = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    X_test = np.column_stack([rng.normal(size=n), rng.normal(loc=4, size=n)])  # 2nd col drifted

    result = dataset_drift(X_train, X_test, feature_names=["stable_feature", "drifted_feature"])

    assert result["available"] is True
    assert result["features"][0]["feature"] == "drifted_feature"
    assert result["features"][0]["status"] == "significant_shift"
    assert result["features"][-1]["status"] == "stable"
    assert result["significant_shift_count"] == 1


def test_dataset_drift_mismatched_feature_counts():
    result = dataset_drift(np.zeros((10, 3)), np.zeros((10, 2)))

    assert result["available"] is False


def test_psi_handles_nan_values():
    expected = np.array([1, 2, 3, 4, 5] * 20 + [np.nan])
    actual = np.array([1, 2, 3, 4, 5] * 20)

    psi = np.asarray(population_stability_index(expected, actual))

    assert np.isfinite(psi)
    assert psi < 0.1  # same underlying distribution once the NaN is dropped


def test_psi_constant_feature_that_shifts_is_not_falsely_stable():
    train_constant = np.full(20, 1.0)
    test_shifted = np.full(20, 100.0)

    psi = population_stability_index(train_constant, test_shifted)

    assert psi >= 0.25  # must NOT be reported as stable


def test_psi_constant_feature_that_does_not_shift_is_stable():
    train_constant = np.full(20, 1.0)
    test_same = np.full(20, 1.0)

    psi = population_stability_index(train_constant, test_same)

    assert psi == 0.0


def test_dataset_drift_rejects_non_numeric_features():
    X1 = np.array([["London"], ["Paris"]], dtype=object)
    X2 = np.array([["London"], ["Berlin"]], dtype=object)

    result = dataset_drift(X1, X2)

    assert result["available"] is False


def test_dataset_drift_feature_names_length_mismatch():
    X_train = np.random.default_rng(0).normal(size=(50, 5))
    X_test = np.random.default_rng(1).normal(size=(50, 5))

    result = dataset_drift(X_train, X_test, feature_names=["a", "b"])

    assert result["available"] is False
    assert "feature_names" in result["reason"]


def test_psi_binary_column_with_natural_sampling_noise_is_stable():
    # Regression test: a binary 0/1 column (e.g. one-hot encoded) split
    # randomly into train/test will have slightly different proportions
    # just from sampling -- that is NOT real drift and must not be
    # reported as 'significant_shift'.
    rng = np.random.default_rng(42)
    flag = rng.choice([0, 1], size=891, p=[0.35, 0.65])
    train, test = flag[:712], flag[712:]

    psi = population_stability_index(train, test)

    assert psi < 0.1


def test_psi_binary_column_with_real_shift_is_flagged():
    # A genuine shift in a binary column (e.g. a category nearly disappears)
    # must still be caught, not swallowed by the fix above.
    train = np.array([0] * 700 + [1] * 200)
    test = np.array([0] * 50 + [1] * 250)

    psi = population_stability_index(train, test)

    assert psi >= 0.25


def test_dataset_drift_image_data_uses_channel_summaries():
    rng = np.random.default_rng(1)
    imgs_train = rng.random((100, 16, 16, 3))
    imgs_test = rng.random((50, 16, 16, 3)) + 0.4  # brightness shift

    result = dataset_drift(imgs_train, imgs_test)

    assert result["available"] is True
    assert result["image_summary"] is True
    assert len(result["features"]) == 6  # 3 channels x (mean, std)
    assert all(f["feature"].startswith("channel_") for f in result["features"])
    assert result["features"][0]["status"] == "significant_shift"


def test_dataset_drift_image_data_no_shift_is_stable():
    # Needs a reasonably large sample: PSI's default 10 bins require a few
    # hundred points each to be stable, same as any other continuous feature.
    rng = np.random.default_rng(1)
    imgs_train = rng.normal(0.5, 0.1, (1500, 16, 16, 3))
    imgs_test = rng.normal(0.5, 0.1, (500, 16, 16, 3))

    result = dataset_drift(imgs_train, imgs_test)

    assert result["available"] is True
    assert result["significant_shift_count"] == 0


def test_dataset_drift_grayscale_images_without_channel_dim():
    rng = np.random.default_rng(1)
    imgs_train = rng.normal(0.5, 0.1, (100, 16, 16))  # (N, H, W), no channel axis
    imgs_test = rng.normal(0.5, 0.1, (50, 16, 16))

    result = dataset_drift(imgs_train, imgs_test)

    assert result["available"] is True
    assert len(result["features"]) == 2  # 1 channel x (mean, std)


def test_dataset_drift_mismatched_image_channels():
    result = dataset_drift(np.random.rand(10, 8, 8, 3), np.random.rand(10, 8, 8, 1))

    assert result["available"] is False


def test_dataset_drift_mismatched_dimensions():
    result = dataset_drift(np.random.rand(10, 8, 8, 3), np.random.rand(10, 64))

    assert result["available"] is False


def test_dataset_drift_empty_split_does_not_raise():
    result = dataset_drift(np.empty((0, 3)), np.random.default_rng(0).normal(size=(50, 3)))

    assert result["available"] is False
