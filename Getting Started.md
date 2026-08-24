# Getting Started with ModelBrief
ModelBrief is a Python library designed to automatically generate structured reports for machine learning experiments. It allows you to analyse a trained machine learning model without changing your existing training workflow. 
To get started, make sure you have **Python 3.10** installed, then install ModelBrief with `pip install modelbrief`. 
If you are contributing to the project, 
clone the repository with `git clone https://github.com/YOUR_USERNAME/modelbrief.git`, 
create a dedicated environment using `conda create -n modelbrief python=3.10`, 
activate it with `conda activate modelbrief`, and install the project in editable mode using `pip install -e .`. 
Once installed, you can use ModelBrief after training your model. For example, after training a scikit-learn model with `model.fit(X_train, y_train)`,
import ModelBrief using `from modelbrief import ModelBrief`, create a report with `report = ModelBrief(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)`, and display it using `report.show()`. 
ModelBrief is designed to support different machine learning tasks, including classification, regression, clustering, computer vision, natural language processing, and other supported workflows.
It uses an adapter-based architecture to work with frameworks such as scikit-learn, PyTorch, TensorFlow, XGBoost, and LightGBM, with framework-specific support provided through dedicated adapters. 
The generated report can include model information, dataset information, performance metrics, prediction results, error analysis, feature importance, visualisations, and other model-specific information where supported. 
ModelBrief's core reporting functionality does not require an API key, while its optional AI explanation functionality may use an external LLM to provide natural-language interpretations of model results. 
You can explore the examples included in the `examples/` directory, such as `classification.py`, `regression.py`, `clustering.py`, `computer_vision.py`, and `pytorch.py`, by running commands such as `python examples/classification.py`. To verify that everything is working correctly, run the project's test suite with `pytest` or `pytest -v`. 
If you encounter an import error, make sure your ModelBrief environment is activated and that the package has been installed with `pip install -e .`.
If a model is not recognised, verify that its framework is supported and that the required dependency is installed. For more information about contributing to ModelBrief, see `CONTRIBUTING.md`; for common questions, see `Frequently Asked Questions.md`; and for the complete project overview, see `README.md`.
