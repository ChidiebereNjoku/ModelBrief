# Regression

ModelBrief supports already-trained regression models that expose a `predict` method, including scikit-learn-compatible estimators and frameworks with a compatible adapter. Numeric target values are evaluated against generated predictions.

A regression report can include standard error and fit measures, residual information, and feature importance when the model exposes supported attributes.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from modelbrief import ModelBrief

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1,
)

model = RandomForestRegressor(random_state=1).fit(X_train, y_train)

report = ModelBrief(
    model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
report.pdf("reports/regression.pdf")
```

See [`examples/regression.py`](../examples/regression.py) for the runnable example, and [`examples/time_series.py`](../examples/time_series.py) for numeric forecasting with `task="time_series"`.
