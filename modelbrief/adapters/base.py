from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class BaseAdapter(ABC):
    def __init__(self, model): self.model=model
    @abstractmethod
    def predict(self, X): raise NotImplementedError
    def predict_proba(self, X):
        if hasattr(self.model,"predict_proba"): return np.asarray(self.model.predict_proba(X))
        return None
    def parameters(self):
        if hasattr(self.model,"get_params"):
            try: return self.model.get_params(deep=False)
            except Exception: pass
        if hasattr(self.model,"count_params"):
            try: return {"trainable_or_total_parameters":int(self.model.count_params())}
            except Exception: pass
        return {"class":type(self.model).__name__}
    def feature_importance(self, feature_names=None):
        values=getattr(self.model,"feature_importances_",None)
        if values is None and hasattr(self.model,"coef_"):
            values=np.asarray(self.model.coef_); values=np.mean(np.abs(values),axis=0) if values.ndim>1 else np.abs(values)
        if values is None: return None
        values=np.asarray(values).reshape(-1); names=feature_names or [f"feature_{i}" for i in range(len(values))]
        return sorted(zip(names,values.tolist()),key=lambda x:abs(x[1]),reverse=True)
