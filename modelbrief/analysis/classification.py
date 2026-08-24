from ..metrics.classification import classification_metrics
def analyse(y,pred,proba=None): return classification_metrics(y,pred,proba)
