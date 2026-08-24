from sklearn.datasets import make_classification,make_regression
def classification_data(): return make_classification(n_samples=120,n_features=6,random_state=2)
def regression_data(): return make_regression(n_samples=120,n_features=6,random_state=2)
