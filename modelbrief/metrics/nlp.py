def token_accuracy(y,pred):
 total=sum(len(a) for a in y); correct=sum(sum(x==z for x,z in zip(a,b)) for a,b in zip(y,pred)); return correct/total if total else 0.0
