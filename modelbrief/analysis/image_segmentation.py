from ..metrics.segmentation import segmentation_metrics
def analyse(y,pred): return segmentation_metrics(y,pred)
