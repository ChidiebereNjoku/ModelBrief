def misclassified_texts(texts,y,pred,n=20): return [{"text":str(t),"actual":str(a),"predicted":str(p)} for t,a,p in zip(texts,y,pred) if a!=p][:n]
