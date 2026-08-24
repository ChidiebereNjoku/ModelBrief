import numpy as np

def regression_metrics(y,pred):
 from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,median_absolute_error,explained_variance_score
 y=np.asarray(y); pred=np.asarray(pred); mse=mean_squared_error(y,pred)
 return {"mae":float(mean_absolute_error(y,pred)),"mse":float(mse),"rmse":float(np.sqrt(mse)),"r2":float(r2_score(y,pred)),"median_ae":float(median_absolute_error(y,pred)),"explained_variance":float(explained_variance_score(y,pred))}
