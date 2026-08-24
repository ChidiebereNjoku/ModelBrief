# Frequently Asked Questions

This document answers common questions about **ModelBrief**, including installation, supported machine learning models, reports, framework adapters, AI explanations, development, and contributions.

---

## What is ModelBrief?

**ModelBrief** is a Python library designed to automatically generate structured reports for machine learning experiments.

Instead of manually collecting information about a model, dataset, performance metrics, errors, feature importance, and other experiment details, ModelBrief brings relevant information together into a single report.

The project is designed around a modular architecture that allows different machine learning frameworks and model types to be supported through dedicated adapters.

---

## What problem does ModelBrief solve?

Machine learning experiments often generate information across notebooks, Python scripts, terminal output, evaluation metrics, and visualisations.

As experiments become more complex, it can be difficult to keep track of:

- Which model was used
- What dataset was used
- How the model performed
- Which metrics were important
- Where the model made errors
- Which features influenced predictions
- How different experiments compare
- How to communicate results to other people

ModelBrief aims to automate much of this reporting process.

The goal is to make machine learning experiment reporting:

- Faster
- More consistent
- Easier to understand
- Easier to reproduce
- Easier to communicate

---

## Who is ModelBrief for?

ModelBrief is designed for:

- Data scientists
- Machine learning engineers
- AI engineers
- Researchers
- Students
- Machine learning practitioners
- Analysts
- Teams documenting ML experiments

It can be particularly useful when working on multiple experiments and needing a consistent way to document model results.

---

## What can ModelBrief analyse?

ModelBrief is designed to support different machine learning tasks and model families.

These include:

- Classification
- Regression
- Clustering
- Deep learning
- Computer vision
- Natural language processing
- Time-series modelling
- Tree-based models
- Other supported machine learning models

The exact analysis available depends on the model type, framework, adapter, and data provided.

---

## Which machine learning frameworks are supported?

ModelBrief is designed around an adapter architecture that can support multiple frameworks.

The project is intended to support frameworks such as:

- Scikit-learn
- XGBoost
- LightGBM
- PyTorch
- TensorFlow

Additional model types and frameworks can be added through the adapter system.

---

## Does ModelBrief support classification models?

Yes.

ModelBrief is designed to analyse classification models and provide relevant information such as:

- Model information
- Dataset information
- Classification performance
- Evaluation metrics
- Prediction errors
- Feature importance where supported
- Visualisations where applicable

Examples of classification models include:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machines
- Neural Networks
- Other supported classifiers

---

## Does ModelBrief support regression models?

Yes.

Regression models can be analysed using the same general ModelBrief reporting workflow.

Depending on the model and available analysis components, reports may include:

- Model information
- Dataset information
- Regression metrics
- Prediction errors
- Feature importance
- Visualisations

Examples include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost Regression
- Other supported regression models

---

## Does ModelBrief support clustering models?

ModelBrief is designed to support unsupervised learning workflows such as clustering.

Examples include:

- K-Means
- Hierarchical clustering
- DBSCAN
- Other supported clustering algorithms

The available report information depends on the clustering algorithm and the information supplied to ModelBrief.

---

## Does ModelBrief support XGBoost and LightGBM?

ModelBrief is designed to support gradient-boosting frameworks through dedicated adapters.

This allows framework-specific logic to remain separate from the ModelBrief core.

Potential supported functionality includes:

- Model information
- Dataset information
- Performance metrics
- Feature importance
- Predictions
- Error analysis

The exact functionality depends on the adapter available in the installed version.

---

## Does ModelBrief support PyTorch and TensorFlow?

ModelBrief is designed to support deep learning frameworks through dedicated adapters.

These include:

- PyTorch
- TensorFlow

Deep learning reports may contain model-specific information such as:

- Model architecture
- Model parameters
- Dataset information
- Evaluation results
- Performance information
- Other supported model details

The available analysis depends on the model and adapter.

---

## Does ModelBrief support NLP models?

ModelBrief is designed to support Natural Language Processing models where an appropriate adapter is available.

Potential NLP use cases include:

- Text classification
- Sentiment analysis
- Transformer models
- Text embeddings
- Language models
- NLP pipelines

The available report information depends on the model, task, and adapter.

---

## Does ModelBrief support computer vision models?

Yes, where the relevant framework and adapter are supported.

Potential use cases include:

- Image classification
- CNN models
- Transfer learning
- Deep learning image models
- Other supported computer vision workflows

Model-specific analysis depends on the framework and model being used.

---

## Does ModelBrief support time-series models?

ModelBrief is intended to support time-series and forecasting models as the relevant analysis and adapters are implemented.

Potential model types include:

- ARIMA
- SARIMA
- Exponential Smoothing
- Other statistical forecasting models
- Machine learning forecasting models

Support depends on the current implementation of the relevant adapter.

---

## How do I install ModelBrief?

ModelBrief is available on PyPI, install it using:

```bash
pip install modelbrief