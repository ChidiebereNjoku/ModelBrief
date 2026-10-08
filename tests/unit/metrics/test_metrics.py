from modelbrief.metrics.classification import classification_metrics
from modelbrief.metrics.regression import regression_metrics
def test_metrics():
 assert classification_metrics([0,1],[0,1])["accuracy"]==1.0
 assert regression_metrics([1,2],[1,2])["rmse"]==0.0
