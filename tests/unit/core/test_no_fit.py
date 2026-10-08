from modelbrief import ModelBrief


class M:
    """Test classifier that raises an error if ModelBrief attempts training."""

    _estimator_type = "classifier"

    def fit(self, *args, **kwargs):
        raise AssertionError(
            "ModelBrief must not retrain the supplied model"
        )

    def predict(self, X):
        # Return both classes to avoid the single-label confusion-matrix warning.
        return [0, 1][:len(X)]


def test_never_fits():
    result = ModelBrief(
        model=M(),
        X_test=[[1], [2]],
        y_test=[0, 1],
    ).analyse()

    assert result.metadata["task"] == "classification"

    performance = result.section("MODEL PERFORMANCE")
    assert performance is not None

    assert performance.content["test"]["accuracy"] == 1.0