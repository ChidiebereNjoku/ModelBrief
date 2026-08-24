import numpy as np
from sklearn.linear_model import LinearRegression
from modelbrief import ModelBrief
t=np.arange(100); X=t.reshape(-1,1); y=.4*t+np.sin(t/4); model=LinearRegression().fit(X[:80],y[:80]); ModelBrief(model,X_test=X[80:],y_test=y[80:],task="time_series").html("reports/time_series.html")
