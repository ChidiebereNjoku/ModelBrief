# Classification

ModelBrief evaluates already-trained classification models by comparing their predictions with a supplied target. It supports scikit-learn-compatible classifiers, PyTorch and TensorFlow/Keras classifiers, and Transformers pipelines used for classification tasks.

When probability predictions are available, ModelBrief can include log loss and ROC AUC in the report. Classification reports also include weighted precision, recall, and F1 scores, accuracy, balanced accuracy, and a confusion matrix.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from modelbrief import ModelBrief

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1,
    stratify=y,
)

model = RandomForestClassifier(random_state=1).fit(X_train, y_train)

report = ModelBrief(
    model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
report.show()
```

For a complete runnable example, see [`examples/classification.py`](../examples/classification.py). Additional framework examples are available in [`examples/pytorch.py`](../examples/pytorch.py), [`examples/tensorflow_example.py`](../examples/tensorflow_example.py), and [`examples/nlp/text_classification.py`](../examples/nlp/text_classification.py).
