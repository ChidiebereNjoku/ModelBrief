import pytest

pytest.importorskip("tensorflow")

import numpy as np
from modelbrief import ModelBrief


def _make_images(rng, n, brightness_shift=0.0, size=16):
    X = rng.normal(0.3 + brightness_shift, 0.05, size=(n, size, size, 1)).astype("float32")
    y = rng.integers(0, 2, size=n)
    return np.clip(X, 0, 1), y


def _build_tiny_cnn():
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(16, 16, 1)),
        layers.Conv2D(4, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def test_cnn_drift_is_stable_with_no_real_shift():
    # Larger sample than the other CNN test: PSI needs enough images per bin
    # to not mistake ordinary sampling noise for drift.
    rng = np.random.default_rng(1)
    X_train, y_train = _make_images(rng, 1200)
    X_test, y_test = _make_images(rng, 400)

    model = _build_tiny_cnn()
    model.fit(X_train, y_train, epochs=1, verbose=0)

    drift = ModelBrief(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        task="computer_vision", ai=False).data_drift()
    assert drift["available"] is True
    assert drift["image_summary"] is True
    assert drift["significant_shift_count"] == 0


def test_cnn_drift_catches_a_real_brightness_shift():
    rng = np.random.default_rng(1)
    X_train, y_train = _make_images(rng, 100)
    X_test, y_test = _make_images(rng, 40, brightness_shift=0.35)

    model = _build_tiny_cnn()
    model.fit(X_train, y_train, epochs=1, verbose=0)

    drift = ModelBrief(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        task="computer_vision", ai=False).data_drift()
    assert drift["available"] is True
    assert drift["significant_shift_count"] >= 1
    assert drift["features"][0]["feature"] == "channel_0_mean"
