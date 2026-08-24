from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from modelbrief import ModelBrief
X,_=make_blobs(n_samples=200,random_state=1); model=KMeans(3,random_state=1,n_init="auto").fit(X); ModelBrief(model,X_test=X,task="clustering").show()
