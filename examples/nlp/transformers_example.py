from transformers import pipeline
from modelbrief import ModelBrief
model=pipeline("sentiment-analysis"); texts=["I like it","I dislike it"]; labels=["POSITIVE","NEGATIVE"]; ModelBrief(model,X_test=texts,y_test=labels,task="nlp").show()
