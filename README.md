# ModelBrief

**Train your model your way. ModelBrief analyses it and turns the results into a complete model report.**

[![PyPI version](https://img.shields.io/pypi/v/modelbrief.svg)](https://pypi.org/project/modelbrief)
[![Downloads](https://static.pepy.tech/badge/modelbrief)](https://pepy.tech/project/modelbrief)
[![Python Tests](https://github.com/ChidiebereNjoku/ModelBrief/actions/workflows/tests.yml/badge.svg)](https://github.com/ChidiebereNjoku/ModelBrief/actions) 
[![Python Version](https://img.shields.io/pypi/pyversions/modelbrief.svg)](https://www.python.org/downloads)
[![License](https://img.shields.io/pypi/l/modelbrief.svg)]( https://opensource.org/license/Apache-2.0)


ModelBrief accepts an **already-trained** model plus any available train, validation, and test data. It never calls `fit`, never retrains the model, and produces console, standalone HTML, and PDF reports.

## Overview

ModelBrief is an open-source Python library for analysing trained machine-learning models and automatically generating structured reports.
Reports can include:

Model overview and parameters,
Dataset summary,
Task detection,
Feature importance,
Model performance and evaluation metrics,
Confusion matrix,
Error analysis,
Visualisations,
Optional AI-generated explanation and recommendations

## Supported workflows

* Scikit-learn classifiers, regressors, clusters and compatible estimators
* XGBoost, LightGBM and CatBoost through optional extras
* PyTorch and TensorFlow/Keras classification models
* CNN/image classification data, plus extensible detection and segmentation metrics
* NLP pipelines, PyTorch/TensorFlow text models, token and text error utilities
* Regression, classification, clustering, time-series forecast evaluation and anomaly detection

Deep-learning inputs differ by project. Pass model-ready tensors or arrays. For custom data loaders, output schemas, object detection, generation, segmentation, or unusual prediction APIs, subclass `BaseAdapter` and pass `adapter=YourAdapter(model)`.

```python
ModelBrief(model, X_test=X_test, y_test=y_test, task="time_series")
ModelBrief(model, X_test=images, y_test=labels, task="computer_vision")
ModelBrief(model, X_test=texts, y_test=labels, task="nlp")
ModelBrief(model, X_test=X, task="clustering")
```

## Design guarantees

* The supplied model is treated as trained and is never fitted.
* Validation is optional and independently reported when present.
* Optional frameworks are not heavily imported.
* A failed optional analysis is recorded as a warning rather than destroying the whole report.
* HTML is standalone. PDF creation uses ReportLab.

Current version: 0.2.1
Python: 3.10+

## Quick start

```bash
pip install modelbrief
```

```python
from modelbrief import ModelBrief

report = ModelBrief(
    model=model,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,       # optional
    X_test=X_test, y_test=y_test,
    ai=False,
)
print(report.show())
report.pdf("reports/model.pdf")
report.html("reports/model.html")
```

Reports include:
 **Model overview, Dataset, Task, Parameters, Feature importance, Model performance, Confusion matrix, Error analysis, Charts, and Optional AI guidance**.

## Framework-specific dependencies remain optional.

```bash
pip install modelbrief
pip install "modelbrief[xgboost]"
pip install "modelbrief[lightgbm]"
pip install "modelbrief[catboost]"
pip install "modelbrief[pytorch]"       # includes torch and torchvision
pip install "modelbrief[tensorflow]"
pip install "modelbrief[vision]"
pip install "modelbrief[nlp]"           # transformers and torch
pip install "modelbrief[ai]"            # Groq SDK
pip install "modelbrief[all]"           # every integration
```

## Setting Up the Development Environment including pytest
 
```bash
git clone <your-fork-url>
cd modelbrief
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Groq AI, optional

1. Create or retrieve an API key from [GroqCloud](https://console.groq.com/home).
2. Install the optional client: `pip install "modelbrief[ai]"`.
3. Set the key in the environment. Never put it in source code or commit it.

PowerShell:
```powershell
$env:GROQ_API_KEY="your-api-key"
$env:MODELBRIEF_GROQ_MODEL="openai/gpt-oss-20b" # optional
```

Windows CMD:
```cmd
set GROQ_API_KEY=your-api-key
set MODELBRIEF_GROQ_MODEL=openai/gpt-oss-20b
```

macOS/Linux:
```bash
export GROQ_API_KEY="your-api-key"
export MODELBRIEF_GROQ_MODEL="openai/gpt-oss-20b" # optional
```
## Install or Update
Using pip:
pip install --upgrade modelbrief  # Get the latest features!, if you already have modelbrief installed .


## Run tests:
pytest

Then use `ModelBrief(..., ai=True)`. With `ai=False`, Groq is neither imported nor called. AI output is supplementary and should be reviewed by a qualified human.

## contributing,workflow and example
Contributions are welcome! ModelBrief is intended to be an open-source project where developers and data scientists can add adapters, analyses, visualisations, tests and improvements.

See `examples/` for classification, regression, clustering, time series, anomaly detection, PyTorch, TensorFlow, computer vision, CNN, object detection, segmentation, NLP, sentiment, Transformers, and text generation examples.

Fork, clone, create a branch, install `.[dev]`, add code and tests, run `pytest`, and open a pull request. Keep framework-specific dependencies optional. Do not commit secrets, datasets, checkpoints, or large model weights.

## Branching Strategy

- **`main`** → Production-ready code
- **`dev`** → Active development
- **`feature/*`** → New features
- **`fix/*`** → Bug fixes

## Maintainers
ModelBrief is maintained by a data scientist and software developer:


| Name             | GitHub           | Email                                                         |
| ---------------- | ---------------- | ------------------------------------------------------------- |
| Chidiebere Njoku | @ChidiebereNjoku | [chidexnj@gmail.com](mailto:chidexnj@gmail.com)               |  |
| Micheal Adegoro  | @micheal         | [adegboromicheal@gmail.com](mailto:adegboromicheal@gmail.com) |


##  License

Apache License 2.0


