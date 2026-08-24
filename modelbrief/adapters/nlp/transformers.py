import numpy as np
from ..base import BaseAdapter
class TransformersAdapter(BaseAdapter):
    def predict(self,X):
        outputs=self.model(X)
        labels=[]
        for item in outputs:
            if isinstance(item,list): item=max(item,key=lambda z:z.get("score",0))
            labels.append(item.get("label",item) if isinstance(item,dict) else item)
        return np.asarray(labels,dtype=object)
    def parameters(self): return {"pipeline_task":getattr(self.model,"task",None),"model_class":type(getattr(self.model,"model",self.model)).__name__}
