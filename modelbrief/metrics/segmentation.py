import numpy as np
def segmentation_metrics(y,pred):
 y=np.asarray(y).astype(bool); p=np.asarray(pred).astype(bool); inter=np.logical_and(y,p).sum(); union=np.logical_or(y,p).sum(); return {"iou":float(inter/union) if union else 1.0,"dice":float(2*inter/(y.sum()+p.sum())) if y.sum()+p.sum() else 1.0}
