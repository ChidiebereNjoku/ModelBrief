from __future__ import annotations
import base64,io,json
import numpy as np
from .context import ReportContext,DataSplit
from .detector import detect_adapter,detect_task
from .result import ReportResult
from ..analysis.dataset import describe_dataset
from ..analysis.model import model_overview

class ModelBrief:
 def __init__(self,model,X_train=None,X_val=None,X_test=None,y_train=None,y_val=None,y_test=None,task=None,ai=False,feature_names=None,target_names=None,adapter=None,**options):
  self.context=ReportContext(model,DataSplit(X_train,y_train),DataSplit(X_val,y_val),DataSplit(X_test,y_test),task,feature_names,target_names,options=options)
  self.adapter=adapter or detect_adapter(model,X_train if X_train is not None else X_test); self.ai=ai; self._result=None
  print(
    "Selected adapter:",
    type(self.adapter).__name__,
    "| Model:",
    type(model).__module__,
    type(model).__name__,
)
 def _split_analysis(self,name,split,task):
  if split.X is None: return None
  pred=np.asarray(self.adapter.predict(split.X)); out={"samples":len(pred)}
  if split.y is None:
   if task=="clustering":
    from ..metrics.clustering import clustering_metrics; out.update(clustering_metrics(split.X,pred))
   else: out["predictions_generated"]=True
   return out
  if task in ("classification","computer_vision","nlp","text_classification","sentiment"):
   from ..metrics.classification import classification_metrics
   try: proba=self.adapter.predict_proba(split.X)
   except Exception: proba=None
   out.update(classification_metrics(split.y,pred,proba))
  elif task in ("time_series",):
   from ..metrics.time_series import time_series_metrics; out.update(time_series_metrics(split.y,pred))
  elif task in ("anomaly_detection",):
   from ..metrics.anomaly_detection import anomaly_metrics; out.update(anomaly_metrics(split.y,pred))
  elif task in ("image_segmentation","segmentation"):
   from ..metrics.segmentation import segmentation_metrics; out.update(segmentation_metrics(split.y,pred))
  else:
   from ..metrics.regression import regression_metrics; out.update(regression_metrics(split.y,pred))
  return out
 def analyse(self,force=False):
  if self._result is not None and not force: return self._result
  c=self.context; yref=c.train.y if c.train.y is not None else (c.test.y if c.test.y is not None else c.validation.y); task=detect_task(c.model,yref,c.task); r=ReportResult(metadata={"task":task,"library_version":"0.2.0"})
  r.add("MODEL OVERVIEW",model_overview(c.model,self.adapter,task)); r.add("DATASET",{"train":describe_dataset(c.train.X,c.train.y),"validation":describe_dataset(c.validation.X,c.validation.y),"test":describe_dataset(c.test.X,c.test.y)}); r.add("TASK",{"detected_or_requested":task}); r.add("PARAMETERS",self.adapter.parameters())
  fi=self.adapter.feature_importance(c.feature_names)
  r.add("FEATURE IMPORTANCE",{"available":fi is not None,"ranking":fi[:50] if fi else []})
  perf={};
  for name,split in (("train",c.train),("validation",c.validation),("test",c.test)):
   try:
    value=self._split_analysis(name,split,task)
    if value is not None: perf[name]=value
   except Exception as e: r.warnings.append(f"{name} analysis skipped: {type(e).__name__}: {e}")
  r.add("MODEL PERFORMANCE",perf)
  cm=next((v.get("confusion_matrix") for v in (perf.get("test",{}),perf.get("validation",{}),perf.get("train",{})) if "confusion_matrix" in v),None); r.add("CONFUSION MATRIX",{"available":cm is not None,"matrix":cm})
  r.add("ERROR ANALYSIS",self._errors(task,c.test if c.test.X is not None else c.validation if c.validation.X is not None else c.train))
  self._figures(r,fi,cm,task,c)
  if self.ai:
   from ..ai.providers.groq import GroqProvider
   from ..ai.explainer import explain
   from ..ai.recommender import recommend
   provider=GroqProvider(); r.add("AI EXPLANATION",{"text":explain(r,provider)}); r.recommendations.append(recommend(r,provider))
  self._result=r; return r
 def _errors(self,task,split):
  if split.X is None or split.y is None: return {"available":False}
  try:
   pred=np.asarray(self.adapter.predict(split.X)); y=np.asarray(split.y)
   if task in ("classification","computer_vision","nlp","text_classification","sentiment","anomaly_detection"):
    idx=np.flatnonzero(pred!=y)[:25]; return {"available":True,"error_count":int(np.sum(pred!=y)),"examples":[{"index":int(i),"actual":str(y[i]),"predicted":str(pred[i])} for i in idx]}
   err=np.abs(y-pred); idx=np.argsort(err.reshape(-1))[::-1][:25]; return {"available":True,"mean_absolute_error":float(np.mean(err)),"largest":[{"index":int(i),"actual":float(y.reshape(-1)[i]),"predicted":float(pred.reshape(-1)[i]),"absolute_error":float(err.reshape(-1)[i])} for i in idx]}
  except Exception as e: return {"available":False,"reason":str(e)}
 def _add_fig(self,r,fig,title):
  b=io.BytesIO(); fig.savefig(b,format="png",dpi=130,bbox_inches="tight"); r.figures.append({"title":title,"data":base64.b64encode(b.getvalue()).decode()});
  import matplotlib.pyplot as plt; plt.close(fig)
 def _figures(self,r,fi,cm,task,c):
  try:
   if fi:
    from ..visualization.feature_importance import plot_feature_importance; self._add_fig(r,plot_feature_importance(fi),"Feature importance")
   if cm:
    from ..visualization.confusion_matrix import plot_confusion_matrix; self._add_fig(r,plot_confusion_matrix(cm,c.target_names),"Confusion matrix")
   split=c.test if c.test.X is not None else c.validation
   if task in ("regression","time_series") and split.X is not None and split.y is not None:
    from ..visualization.residuals import plot_residuals; self._add_fig(r,plot_residuals(split.y,self.adapter.predict(split.X)),"Residuals")
  except Exception as e: r.warnings.append(f"Visualisation skipped: {e}")
 def show(self):
  from ..output.console import render_console
  text=render_console(self.analyse()); print(text); return text
 def html(self,path="modelbrief_report.html"):
  from ..output.html import render_html; return render_html(self.analyse(),path)
 def pdf(self,path="modelbrief_report.pdf"):
  from ..output.pdf import render_pdf; return render_pdf(self.analyse(),path)
 @property
 def result(self): return self.analyse()
