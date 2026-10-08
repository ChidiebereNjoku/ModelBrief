def model_overview(model,adapter,task):
 return {"model_class":type(model).__name__,"module":type(model).__module__,"task":task,"fitted_model_supplied":True,"retraining_performed":False,"parameters":adapter.parameters()}
