def analyse(X,embedded):
 return {"input_dimensions":getattr(X,"shape",[None,None])[-1],"output_dimensions":getattr(embedded,"shape",[None,None])[-1],"samples":len(embedded)}
