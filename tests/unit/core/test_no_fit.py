from modelbrief import ModelBrief
class M:
 _estimator_type="classifier"
 def fit(self,*a): raise AssertionError("must not fit")
 def predict(self,X): return [0]*len(X)
def test_never_fits(): assert ModelBrief(M(),X_test=[[1],[2]],y_test=[0,0]).analyse().metadata["task"]=="classification"
