import numpy as np

def describe_dataset(X,y=None):
    if X is None: return {"available":False}
    shape=getattr(X,"shape",None) or (len(X),)
    info={"available":True,"rows":int(shape[0]),"shape":tuple(int(i) for i in shape),"type":type(X).__name__}
    try:
        a=np.asarray(X); info["missing_values"]=int(np.sum(a!=a)) if a.dtype.kind in "fc" else None
    except Exception: info["missing_values"]=None
    if y is not None:
        ya=np.asarray(y); info["target_shape"]=tuple(ya.shape); info["target_unique"]=int(len(np.unique(ya))) if ya.size and ya.ndim==1 else None
    return info
