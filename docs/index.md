# ModelBrief documentation

Welcome to the ModelBrief documentation. ModelBrief is a Python library that analyses an already-trained machine learning model and generates a structured report from the model, its data, predictions, and evaluation results. ModelBrief does not retrain or fit the supplied model.

Use this page to find the guide that matches your workflow. Reports can contain model and dataset information, performance metrics, feature importance, prediction errors, charts, and optional AI-generated guidance.

## Start here

- [Getting started](getting_started.md) explains how to install ModelBrief and create your first report.
- [Supported models](supported_models.md) lists the framework integrations, adapters, and analysis tasks available in this release.
- [Frequently asked questions](../Frequently%20Asked%20Questions.md) covers common installation, reporting, framework, and usage questions.
- [Contributing](contributing.md) describes how to contribute improvements and documentation.

## Choose a workflow

| Documentation | Use this when you want to |
|---|---|
| [Classification](classification.md) | Evaluate classifiers and review metrics, predictions, errors, and confusion matrices. |
| [Regression](regression.md) | Generate reports for regression predictions and numeric error analysis. |
| [Computer vision](computer_vision.md) | Work with image-classification and image-based model workflows. |
| [Natural language processing](nlp.md) | Analyse text classification, sentiment, generation, Transformers, or other NLP workflows. |
| [AI features](ai_features.md) | Configure optional AI explanations and recommendations. |

The runnable scripts in [`examples/`](../examples/) demonstrate these workflows for classification, regression, clustering, time series, anomaly detection, PyTorch, TensorFlow, computer vision, NLP, and Transformers. You can run an example from the project root, for example:

```bash
python examples/classification.py
```

## Installation

Install the core package:

```bash
pip install modelbrief
```

Framework-specific integrations are optional. Install only the extra needed for the models you use:

```bash
pip install "modelbrief[xgboost]"
pip install "modelbrief[lightgbm]"
pip install "modelbrief[catboost]"
pip install "modelbrief[pytorch]"
pip install "modelbrief[tensorflow]"
pip install "modelbrief[vision]"
pip install "modelbrief[nlp]"
pip install "modelbrief[ai]"
pip install "modelbrief[all]"
```

See [Supported models](supported_models.md) for framework adapters and task coverage.

## Generate a report

Pass an already-trained model and any available training, validation, or test data to `ModelBrief`. A report can be displayed in the console or exported as standalone HTML or PDF:

```python
from modelbrief import ModelBrief

report = ModelBrief(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
)

report.show()
report.html("reports/model.html")
report.pdf("reports/model.pdf")
```

Console output is useful for quick inspection, HTML is suitable for sharing an interactive standalone report, and PDF provides a document suitable for archiving or distribution.

## Examples and troubleshooting

- Start with the code in [`examples/`](../examples/) that matches your task or framework.
- Use [Supported models](supported_models.md) to confirm that a framework and prediction interface are supported.
- See [Frequently asked questions](../Frequently%20Asked%20Questions.md) for common setup and usage questions.
- Check the [project README](../README.md) for a concise overview and development setup.
- If an import or framework error occurs, confirm that the relevant optional extra is installed in the active Python environment.
- If a model is not detected, confirm its framework and module, provide an explicit task, or use a custom adapter for a non-standard prediction interface.

## Contribute or ask for help

Contributions are welcome. Read [CONTRIBUTING.md](../CONTRIBUTING.md) before opening an issue or pull request, and use the project issue tracker to report bugs or request documentation improvements.

Do not include API keys, secrets, datasets, model weights, or private information in issues, examples, or pull requests.
