def analyse_texts(texts):
 lengths=[len(str(t).split()) for t in texts]; return {"documents":len(lengths),"mean_tokens":sum(lengths)/len(lengths) if lengths else 0,"max_tokens":max(lengths,default=0)}
