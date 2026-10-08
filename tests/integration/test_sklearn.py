from sklearn.datasets import make_classification,make_regression
from sklearn.linear_model import LogisticRegression,LinearRegression
from modelbrief import ModelBrief

def test_classification_outputs(tmp_path):
 X,y=make_classification(n_samples=100,n_features=5,random_state=1); m=LogisticRegression().fit(X[:70],y[:70]); r=ModelBrief(m,X_train=X[:70],y_train=y[:70],X_val=X[70:85],y_val=y[70:85],X_test=X[85:],y_test=y[85:])
 assert "MODEL OVERVIEW" in r.show(); assert r.html(tmp_path/"r.html").exists(); assert r.pdf(tmp_path/"r.pdf").exists()
def test_regression():
 X,y=make_regression(n_samples=60,n_features=4,random_state=1); m=LinearRegression().fit(X[:40],y[:40]); result=ModelBrief(m,X_test=X[40:],y_test=y[40:]).analyse(); assert result.metadata["task"]=="regression"
