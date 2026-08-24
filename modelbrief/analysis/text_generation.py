def analyse(generated):
 texts=[str(x) for x in generated]; return {"generations":len(texts),"mean_characters":sum(map(len,texts))/len(texts) if texts else 0}
