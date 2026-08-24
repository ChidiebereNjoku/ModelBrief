from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from modelbrief import ModelBrief
X,y=load_diabetes(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=1); model=RandomForestRegressor(random_state=1).fit(Xtr,ytr); ModelBrief(model,X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte).pdf("reports/regression.pdf")
