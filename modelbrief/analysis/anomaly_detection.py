from ..metrics.anomaly_detection import anomaly_metrics
def analyse(y,pred,scores=None): return anomaly_metrics(y,pred,scores)
