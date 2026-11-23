# Supported models

ModelBrief analyses already-trained models and supplied data. It does not fit or retrain models. The model, its prediction interface, and the data format determine whether a built-in adapter can be used directly.

Most supported workflows accept NumPy-compatible arrays. Tabular models also accept compatible tabular objects, such as pandas DataFrames, when the model supports them. Framework-specific models may require tensors or text inputs in the format expected by the trained model.

## Frameworks

| Framework or model type | Adapter | Installation |
|---|---|---|
| scikit-learn classifiers, regressors, clustering estimators, and pipelines | `SklearnAdapter` | Included with ModelBrief |
| XGBoost estimators | `XGBoostAdapter` | `pip install "modelbrief[xgboost]"` |
| LightGBM estimators | `LightGBMAdapter` | `pip install "modelbrief[lightgbm]"` |
| CatBoost estimators | `CatBoostAdapter` | `pip install "modelbrief[catboost]"` |
| PyTorch models | `PyTorchAdapter` | `pip install "modelbrief[pytorch]"` |
| TensorFlow and Keras models | `TensorFlowAdapter` | `pip install "modelbrief[tensorflow]"` |
| Hugging Face Transformers pipelines | `TransformersAdapter` | `pip install "modelbrief[nlp]"` |
| Custom or compatible estimators | `BaseAdapter` or `SklearnAdapter` fallback | Depends on the model implementation |

### scikit-learn and compatible models

scikit-learn estimators and pipelines are supported as already-fitted classifiers, regressors, or clustering models. Their `predict` method is used to produce report predictions, while `predict_proba` is used when available for probability-based metrics.

XGBoost, LightGBM, and CatBoost have dedicated adapter classes that follow the sklearn-style prediction interface. Install the relevant optional extra before analysing models from that framework.

### Deep-learning and NLP models

PyTorch models are evaluated in evaluation mode. NumPy arrays passed to the PyTorch adapter are converted to floating-point tensors, and batched prediction is supported. TensorFlow and Keras models are called through their prediction API.

Hugging Face Transformers pipelines are supported for text workflows. The supplied text should match the pipeline's expected input, and the resulting labels and scores depend on the selected pipeline and model.

## Tasks

ModelBrief supports analysis for:

- Classification
- Regression
- Clustering
- Time-series forecasting
- Anomaly detection
- Computer vision
- NLP and text classification
- Sentiment analysis
- Image segmentation

A task determines which metrics and report sections are produced. Classification metrics include probability information when the adapter provides it, while regression metrics are applied to numeric predictions. Unsupervised clustering and anomaly-detection workflows can also be evaluated with suitable supplied data.

Computer-vision and NLP inputs may need additional preparation outside ModelBrief. Pass model-ready images or text rather than raw files or unprocessed datasets.

For specialised prediction APIs or unsupported model types, provide a custom adapter using `BaseAdapter`. This is useful when a model uses a custom data loader, a non-standard method name, or an unusual output schema.

## Automatic detection

ModelBrief selects an adapter from the model class module:

- `sklearn.*` and `sklearn.pipeline` → scikit-learn
- `transformers.pipelines.*` → Transformers
- `torch*` → PyTorch
- `tensorflow*` or `keras*` → TensorFlow/Keras
- Any other model → scikit-learn-compatible fallback

When `task` is not supplied, ModelBrief uses the estimator type, target values, or available labels to detect the task. Supply `task=` explicitly when detection could be ambiguous, especially for forecasting, NLP, computer vision, segmentation, anomaly detection, or clustering.

## Input and prediction requirements

The supplied model is not retrained and should already contain the parameters and preprocessing behaviour needed for analysis. If training and evaluation require separate preprocessing, apply that preprocessing before passing data to `ModelBrief`.

A model does not need to support every optional operation, such as probability prediction or feature importance. Reporting will include the results that can be produced and may omit or mark unavailable information when the model's interface does not expose it.
