def clustering_metrics(X,labels):
 from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
 n=len(set(labels)); out={"clusters":n}
 if n>1 and n<len(labels):
  out.update(silhouette=float(silhouette_score(X,labels)),calinski_harabasz=float(calinski_harabasz_score(X,labels)),davies_bouldin=float(davies_bouldin_score(X,labels)))
 return out
