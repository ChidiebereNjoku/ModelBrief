import numpy as np
from sklearn.ensemble import IsolationForest
from modelbrief import ModelBrief
X=np.random.default_rng(1).normal(size=(100,4)); model=IsolationForest(random_state=1).fit(X); ModelBrief(model,X_test=X,task="clustering").show()
