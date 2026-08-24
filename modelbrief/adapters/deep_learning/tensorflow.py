import numpy as np
from ..base import BaseAdapter
class TensorFlowAdapter(BaseAdapter):
    def raw_predict(self,X): return np.asarray(self.model.predict(X,verbose=0))
    def predict_proba(self,X):
        z=self.raw_predict(X)
        if z.ndim==1 or z.shape[-1]==1: return z.reshape(-1)
        return z
    def predict(self,X):
        p=self.predict_proba(X); return (p>=.5).astype(int) if p.ndim==1 else p.argmax(axis=1)
