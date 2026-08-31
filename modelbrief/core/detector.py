from __future__ import annotations

import numpy as np

from ..adapters.tabular.sklearn import SklearnAdapter


def detect_adapter(model, X=None):
    """Select the correct adapter for the model's framework."""

    module_name = type(model).__module__.lower()

    # Scikit-learn models and pipelines
    if (
        module_name == "sklearn.pipeline"
        or module_name.startswith("sklearn.")
    ):
        return SklearnAdapter(model)

    # Genuine Hugging Face Transformers pipelines
    if (
        module_name == "transformers.pipelines.base"
        or module_name.startswith("transformers.pipelines")
    ):
        from ..adapters.nlp.transformers import TransformersAdapter

        return TransformersAdapter(model)

    # PyTorch models
    if module_name.startswith("torch"):
        from ..adapters.deep_learning.pytorch import PyTorchAdapter

        return PyTorchAdapter(model)

    # TensorFlow and Keras models
    if module_name.startswith(("tensorflow", "keras")):
        from ..adapters.deep_learning.tensorflow import TensorFlowAdapter

        return TensorFlowAdapter(model)

    # Default to the scikit-learn adapter
    return SklearnAdapter(model)


def detect_task(model, y=None, requested=None):
    """Detect the task unless the user explicitly provides one."""

    if requested:
        return requested.lower().strip().replace(" ", "_")

    # Older scikit-learn versions expose estimator type through
    # _estimator_type.
    estimator_type = getattr(model, "_estimator_type", None)

    # Newer scikit-learn versions expose estimator type through
    # __sklearn_tags__().
    if estimator_type is None and hasattr(model, "__sklearn_tags__"):
        try:
            estimator_type = model.__sklearn_tags__().estimator_type
        except (AttributeError, TypeError):
            estimator_type = None

    if estimator_type == "classifier":
        return "classification"

    if estimator_type == "regressor":
        return "regression"

    if estimator_type == "clusterer" or hasattr(model, "labels_"):
        return "clustering"

    if y is None:
        return "clustering"

    target = np.asarray(y)

    if target.dtype.kind in "OUSb":
        return "classification"

    unique_values = len(np.unique(target))

    classification_limit = max(
        20,
        int(np.sqrt(max(len(target), 1))),
    )

    if unique_values <= classification_limit:
        return "classification"

    return "regression"