import numpy as np

from ..base import BaseAdapter


class SklearnAdapter(BaseAdapter):
    """Adapter for scikit-learn and compatible estimators."""

    def predict(self, X):
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))

        return None

