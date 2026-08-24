import numpy as np

def time_series_metrics(y,pred):
 from .regression import regression_metrics
 out=regression_metrics(y,pred); y=np.asarray(y); p=np.asarray(pred); nz=np.abs(y)>1e-12
 out["mape"]=float(np.mean(np.abs((y[nz]-p[nz])/y[nz]))*100) if nz.any() else None
 out["smape"]=float(np.mean(2*np.abs(p-y)/(np.abs(y)+np.abs(p)+1e-12))*100); return out
