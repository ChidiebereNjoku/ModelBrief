from collections import Counter
def most_common_tokens(texts,n=20): return Counter(tok.lower() for t in texts for tok in str(t).split()).most_common(n)
