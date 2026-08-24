from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from modelbrief import ModelBrief
X,y=load_iris(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=1,stratify=y); model=RandomForestClassifier(random_state=1).fit(Xtr,ytr); ModelBrief(model,X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,feature_names=load_iris().feature_names).show()
