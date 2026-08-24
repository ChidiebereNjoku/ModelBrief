from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from modelbrief import ModelBrief
X=["great product","bad service","excellent","awful"]*5; y=[1,0,1,0]*5; m=make_pipeline(TfidfVectorizer(),LogisticRegression()).fit(X,y); ModelBrief(m,X_test=X,y_test=y,task="nlp").show()
