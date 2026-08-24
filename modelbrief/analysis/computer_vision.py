from ..metrics.vision import image_classification_metrics
def analyse(y,pred,proba=None): return image_classification_metrics(y,pred,proba)
