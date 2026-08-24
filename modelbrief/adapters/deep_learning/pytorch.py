from __future__ import annotations
import numpy as np
from ..base import BaseAdapter
class PyTorchAdapter(BaseAdapter):
    def __init__(self,model,device=None,batch_size=64): super().__init__(model); self.device=device; self.batch_size=batch_size
    def _tensor(self,X):
        import torch
        if isinstance(X,torch.Tensor): return X
        return torch.as_tensor(np.asarray(X),dtype=torch.float32)
    def raw_predict(self,X):
        import torch
        self.model.eval(); device=self.device or next(self.model.parameters(),torch.empty(0)).device
        out=[]; t=self._tensor(X)
        with torch.no_grad():
            for batch in torch.split(t,self.batch_size):
                y=self.model(batch.to(device)); y=y.logits if hasattr(y,"logits") else y; out.append(y.detach().cpu())
        return torch.cat(out).numpy()
    def predict_proba(self,X):
        z=self.raw_predict(X)
        if z.ndim==1 or z.shape[-1]==1: return 1/(1+np.exp(-z.reshape(-1)))
        z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)
    def predict(self,X):
        p=self.predict_proba(X); return (p>=.5).astype(int) if p.ndim==1 else p.argmax(axis=1)
    def parameters(self):
        total=sum(p.numel() for p in self.model.parameters()); trainable=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_parameters":total,"trainable_parameters":trainable,"architecture":str(self.model)}
