# Natural language processing

ModelBrief supports text workflows with scikit-learn pipelines, PyTorch or TensorFlow text models, and Hugging Face Transformers pipelines. Supply model-ready text or numeric representations, targets, and an appropriate task such as `nlp`, `text_classification`, or `sentiment`.

A text-classification pipeline can combine preprocessing and a trained classifier:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from modelbrief import ModelBrief

texts = ["great product", "bad service", "excellent", "awful"] * 5
labels = [1, 0, 1, 0] * 5

model = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, labels)

report = ModelBrief(model, X_test=texts, y_test=labels, task="nlp")
report.show()
```

Hugging Face pipelines can be passed directly when they are genuine `transformers.pipelines.*` objects. Text generation and sentiment workflows use their corresponding inputs and analysis helpers.

Repository examples:

- [`examples/nlp/text_classification.py`](../examples/nlp/text_classification.py) — TF-IDF text classification.
- [`examples/nlp/transformers_example.py`](../examples/nlp/transformers_example.py) — Hugging Face sentiment-analysis pipeline.
- [`examples/nlp/sentiment_analysis.py`](../examples/nlp/sentiment_analysis.py) — sentiment example entry point.
- [`examples/nlp/text_generation.py`](../examples/nlp/text_generation.py) — analysis of already-generated text.
- [`examples/nlp/transformers_example.py`](../examples/nlp/transformers_example.py) — Transformers pipeline usage.
