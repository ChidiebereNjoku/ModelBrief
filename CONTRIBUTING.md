# Contributing to ModelBrief

Thank you for your interest in contributing to **ModelBrief**! 🎉

ModelBrief is an open-source Python library designed to automatically generate structured reports for machine learning experiments.

We welcome contributions of all kinds, including:

* Bug fixes
* New model and framework support
* New analysis capabilities
* Improvements to model adapters
* Documentation improvements
* Test improvements
* Performance improvements
* Examples and tutorials
* AI explanation improvements

Whether you are an experienced machine learning engineer or making your first open-source contribution, your contribution is welcome.

---

## Code of Conduct

Please be respectful, constructive, and inclusive when participating in the ModelBrief project.

We are building an environment where contributors can:

* Share ideas
* Ask questions
* Learn from each other
* Review code constructively
* Improve the project together

Technical disagreements are welcome, but discussions should remain professional and respectful.

---

# Getting Started

## 1. Fork and Clone

Fork the ModelBrief repository on GitHub and clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/modelbrief.git
cd modelbrief
```

Replace the repository URL above with the official ModelBrief repository URL if it differs.

Add the original repository as an upstream remote:

```bash
git remote add upstream https://github.com/ORGANIZATION/modelbrief.git
```

You can verify your remotes with:

```bash
git remote -v
```

---

## 2. Set Up the Development Environment

ModelBrief currently targets **Python 3.10**.

### Using Conda

Create a dedicated Conda environment:

```bash
conda create -n modelbrief python=3.10
```

Activate it:

```bash
conda activate modelbrief
```

Verify the Python version:

```bash
python --version
```

---

### Using Python Virtual Environment

Alternatively, create a standard Python virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

---

## 3. Install ModelBrief

Install the project in editable mode:

```bash
pip install -e .
```

Editable installation allows changes to the source code to be reflected immediately without reinstalling the package.

If the project provides development dependencies, install those as specified in the project's package configuration.

---

## 4. Verify the Installation

After installation, verify that ModelBrief can be imported:

```bash
python -c "import modelbrief; print('ModelBrief installed successfully')"
```

If this command runs successfully, your development environment is ready.

---

# Development Workflow

## Branching Strategy

ModelBrief uses the following branch structure:

| Branch                   | Purpose               |
| ------------------------ | --------------------- |
| `main`                   | Production-ready code |
| `dev`                    | Active development    |
| `feature/<feature-name>` | New features          |
| `fix/<bug-name>`         | Bug fixes             |
| `docs/<doc-name>`        | Documentation updates |
| `test/<test-name>`       | Test improvements     |

The general development flow is:

```text
feature/* / fix/* / docs/* / test/*
                 ↓
                dev
                 ↓
               main
```

The `main` branch should contain production-ready code.

The `dev` branch is used for active development and integration of new contributions.

---

## Creating a Feature Branch

First make sure your local `dev` branch is up to date:

```bash
git checkout dev
git pull origin dev
```

Create your feature branch:

```bash
git checkout -b feature/your-feature-name
```

For example:

```bash
git checkout -b feature/add-xgboost-adapter
```

---

## Creating a Bug-Fix Branch

For bug fixes:

```bash
git checkout dev
git pull origin dev
git checkout -b fix/model-detection-error
```

---

## Creating a Documentation Branch

For documentation-only changes:

```bash
git checkout dev
git pull origin dev
git checkout -b docs/update-getting-started
```

---

## Making Changes

When working on a contribution:

1. Create the appropriate branch.
2. Understand the existing implementation.
3. Make your changes.
4. Add or update tests.
5. Run the test suite.
6. Run relevant examples.
7. Update documentation if necessary.
8. Review your changes.
9. Commit your work.
10. Push your branch.
11. Open a Pull Request.

Try to keep each Pull Request focused on one feature, bug, or related group of changes.

---

# Project Structure

ModelBrief uses a modular architecture built around a central reporting system and framework-specific adapters.

A typical project structure is:

```text
modelbrief/
│
├── modelbrief/
│   │
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── report.py
│   │   ├── detector.py
│   │   ├── result.py
│   │   └── context.py
│   │
│   ├── adapters/
│   │   ├── base.py
│   │   ├── sklearn/
│   │   ├── pytorch/
│   │   ├── tensorflow/
│   │   ├── xgboost/
│   │   └── lightgbm/
│   │
│   ├── analysis/
│   │   ├── classification.py
│   │   ├── regression.py
│   │   ├── clustering.py
│   │   ├── errors.py
│   │   └── feature_importance.py
│   │
│   ├── visualization/
│   │
│   └── ai/
│
├── examples/
│   ├── classification.py
│   ├── clustering.py
│   ├── computer_vision.py
│   ├── pytorch.py
│   └── regression.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── README.md
├── CONTRIBUTING.md
├── Frequently Asked Questions.md
├── Getting Started.md
└── pyproject.toml
```

The exact structure may change as ModelBrief develops.

---

# Coding Standards

## Python

ModelBrief follows standard Python best practices and aims to maintain clean, readable, and maintainable code.

Contributors should:

* Follow PEP 8 where practical.
* Use meaningful variable and function names.
* Add type hints where appropriate.
* Write clear docstrings.
* Keep functions focused.
* Avoid unnecessary duplication.
* Keep framework-specific logic inside adapters.
* Avoid unnecessary dependencies.
* Handle errors clearly.

```

## Docstrings

Public classes, functions, and important methods should have useful docstrings.

Document:

* Purpose
* Parameters
* Return values
* Exceptions where relevant

---

# Testing Requirements

**All new functionality should include appropriate tests.**

ModelBrief uses `pytest`.

---

## Running Tests

Run the complete test suite:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_core.py -v
```

Run a specific test:

```bash
pytest tests/test_core.py::test_model_detection -v
```

---


## Unit Tests

Unit tests should test individual components in isolation.

For example:

```text
tests/
└── unit/
    ├── test_detector.py
    ├── test_result.py
    └── test_context.py
```

---

## Integration Tests

Integration tests should verify that multiple ModelBrief components work correctly together.

For example:

```text
tests/
└── integration/
    └── test_import.py
```

An integration test may verify that:

```text
Model
  ↓
Detector
  ↓
Adapter
  ↓
Analysis
  ↓
Report
```

works as expected.

---

# Test Coverage

Contributors should add tests for new functionality whenever practical.

Before opening a Pull Request, check that:

* Existing tests pass.
* New functionality has appropriate tests.
* Existing behaviour has not unintentionally been broken.

You can use `pytest-cov` if coverage reporting is configured:

```bash
pytest --cov=modelbrief
```

For a terminal report showing missing lines:

```bash
pytest --cov=modelbrief --cov-report=term-missing
```

---

# Pre-commit Checks

Before committing changes, review the project for:

* Test failures
* Unused imports
* Debugging statements
* Formatting problems
* Accidental files
* Secrets or credentials
* Unnecessary dependencies

Run:

```bash
pytest -v
```

Check Git status:

```bash
git status
```

Review the actual changes:

```bash
git diff
```

---

# Pull Request Process

## 1. Ensure Your Branch Is Ready

Before opening a Pull Request:

* All relevant tests pass.
* New functionality has appropriate tests.
* Documentation has been updated where necessary.
* No API keys or credentials have been committed.
* Your branch is based on the latest `dev`.
* Your changes are focused and understandable.

Update your branch if necessary:

```bash
git checkout dev
git pull origin dev
```

Then update your feature branch appropriately.

---

## 2. Push Your Branch

For example:

```bash
git push origin feature/add-xgboost-adapter
```

---

## 3. Create the Pull Request

Open a Pull Request against:

```text
dev
```

Do not normally open feature Pull Requests directly against `main`.

The normal flow is:

```text
feature/add-xgboost-adapter
              ↓
             dev
              ↓
            main
```

---

## 4. Pull Request Description

A good Pull Request should explain:

### Summary

What does this change do?

### Motivation

Why is the change needed?

### Changes

What files or components were changed?

### Testing

How was the change tested?

Example:

```markdown
## Summary

Add XGBoost model support to ModelBrief.

## Changes

- Add XGBoost adapter
- Update model detection
- Add unit tests
- Add example
- Update documentation

## Testing

pytest -v

All tests passed.
```

---

## 5. PR Review Process

Maintainers may:

* Review the implementation.
* Run the tests.
* Test the branch locally.
* Request additional tests.
* Request documentation updates.
* Ask questions about implementation decisions.

Contributors should address review comments before the Pull Request is merged.

Please keep review discussions constructive and focused on improving the project.

---

## 6. After Approval

Once the Pull Request has been approved:

* Ensure the final tests pass.
* Resolve any remaining review comments.
* Follow the maintainer's merge instructions.
* The maintainer will merge the contribution into `dev`.

Changes in `dev` can later be included in a production release through `main`.

---

# Commit Message Guidelines

ModelBrief follows a simple, descriptive commit convention.

Where practical, use a format similar to:

```text
<type>(<scope>): <description>
```

---

## Commit Types

| Type       | Purpose                  |
| ---------- | ------------------------ |
| `feat`     | New feature              |
| `fix`      | Bug fix                  |
| `docs`     | Documentation changes    |
| `test`     | Test changes             |
| `refactor` | Code restructuring       |
| `perf`     | Performance improvements |
| `build`    | Build/package changes    |
| `ci`       | CI/CD changes            |
| `chore`    | Maintenance              |

---

## Examples

Good commit messages:

```text
feat(adapters): add XGBoost adapter
```

```text
fix(detector): handle unsupported model types
```

```text
docs(readme): update installation instructions
```

```text
test(core): add report generation tests
```

```text
refactor(report): simplify report generation
```

Avoid vague messages such as:

```text
update
```

```text
changes
```

```text
fix
```

```text
stuff
```

---

## Commit Message Best Practices

* Use the imperative form.
* Keep the subject concise.
* Explain the purpose of the change.
* Keep unrelated changes out of the commit.
* Reference an issue where appropriate.

For example:

```text
feat(adapters): add LightGBM model support
```

is clearer than:

```text
updated files
```

---

# Issue Guidelines

Before opening an issue:

1. Search existing issues.
2. Check whether the issue has already been fixed in `dev`.
3. Make sure you are using a supported version.
4. Gather the information needed to reproduce the issue.

---

## Bug Reports

A useful bug report should include:

* Clear description of the problem
* ModelBrief version
* Python version
* Operating system
* ML framework and version
* Steps to reproduce
* Expected behaviour
* Actual behaviour
* Full error traceback where applicable
* Minimal reproducible example

Example:

````markdown
## Bug Description

ModelBrief fails to detect a supported scikit-learn model.

## Environment

Python: 3.10
ModelBrief: 0.x.x
scikit-learn: x.x.x
OS: Windows

## Expected Behaviour

ModelBrief should automatically select the scikit-learn adapter.

## Actual Behaviour

An exception is raised during adapter detection.

## Reproduction

```python
# Minimal example here
````

````

---

## Feature Requests

Feature requests should explain:

- The problem you are trying to solve.
- The proposed feature.
- Why the feature would be useful.
- Example usage where possible.
- Any alternative approaches considered.

Example:

```markdown
## Feature Request

### Problem

ModelBrief currently does not support a particular model framework.

### Proposed Solution

Add a dedicated adapter for the framework.

### Example

```python
report = ModelBrief(
    model=model,
    X_train=X_train,
    X_test=X_test,
)
````

### Benefits

This would allow users of the framework to generate ModelBrief reports.

````

---

# Adding Model or Framework Support

One of the most important ways to contribute to ModelBrief is by adding support for new machine learning frameworks or model types.

ModelBrief uses an adapter-based architecture for this purpose.

The general flow is:

```text
New ML Framework
       ↓
Framework Adapter
       ↓
ModelBrief Core
       ↓
Analysis
       ↓
Report
````

---

## When Adding an Adapter

A new adapter should generally:

1. Follow the existing adapter interface.
2. Keep framework-specific imports inside the adapter where possible.
3. Avoid unnecessary dependencies.
4. Handle unsupported models clearly.
5. Extract relevant model information.
6. Connect to the appropriate analysis components.
7. Include tests.
8. Include an example where useful.
9. Update documentation.

---

## Optional Framework Dependencies

Do not make users install every machine learning framework.

For example, adding XGBoost support should not require a user who only uses scikit-learn to install XGBoost.

Framework-specific dependencies should remain optional where appropriate.

---

# Optional AI Features

ModelBrief's core reporting functionality should remain usable without an external LLM.

AI functionality is intended to be an optional layer that can provide natural-language explanations of model results.

For example:

```text
Model Metrics
      ↓
Analysis
      ↓
Optional AI Explanation
      ↓
Human-readable interpretation
```

---

## API Keys

If an AI provider is used, API keys must never be committed to Git.

Use environment variables instead.

For example:

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your-api-key"
```

### macOS/Linux

```bash
export GROQ_API_KEY="your-api-key"
```

Never write:

```python
GROQ_API_KEY = "actual-secret-key"
```

inside source code.

Do not commit:

```text
.env
```

or other files containing credentials.

---

# Development Tips

## Check Your Current Branch

```bash
git branch
```

---

## Check Your Changes

```bash
git status
```

Review differences:

```bash
git diff
```

---

## Update Your Local `dev` Branch

```bash
git checkout dev
git pull origin dev
```

---

## Run Tests After a Change

```bash
pytest -v
```

---

## Stop Tests After the First Failure

```bash
pytest -x
```

---

## Run Tests with More Detailed Output

```bash
pytest -vv
```

---

## Show Print Statements During Tests

```bash
pytest -s
```

---

## Run a Specific Test

```bash
pytest tests/test_core.py::test_model_detection -v
```

---

# Security Guidelines

Never commit sensitive information to the ModelBrief repository.

This includes:

* API keys
* Passwords
* Access tokens
* Private credentials
* Private datasets
* Environment secrets
* Cloud credentials

Before committing, check:

```bash
git status
```

and:

```bash
git diff
```

Make sure sensitive files are included in `.gitignore` where appropriate.

For example:

```text
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
dist/
build/
*.egg-info/
reports/
*.pt
*.pth
*.h5
```

---

# Documentation Contributions

Documentation improvements are highly encouraged.

You can contribute by improving:

* `README.md`
* `Getting Started.md`
* `Frequently Asked Questions.md`
* `CONTRIBUTING.md`
* Examples
* API documentation
* Tutorials


# Getting Help

If you need help:

* Check `README.md`.
* Check `Getting Started.md`.
* Check `Frequently Asked Questions.md`.
* Search existing GitHub issues.
* Open a question/discussion where appropriate.
* Open an issue if you have identified a reproducible bug.

When asking for help, provide enough technical context for others to reproduce or understand the problem.

---

# Recognition

Contributors who make meaningful contributions to ModelBrief may be recognised through:

* GitHub contributor history
* Project documentation
* Release notes
* Relevant project acknowledgements

Thank you for helping make **ModelBrief** better! 

---

## ModelBrief Contribution Workflow

For quick reference:

```text
                    ┌───────────────┐
                    │     main      │
                    │  Production   │
                    └───────▲───────┘
                            │
                            │
                    ┌───────┴───────┐
                    │      dev      │
                    │ Development   │
                    └───────▲───────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             │              │              │
       feature/*         fix/*          docs/*
             │              │              │
             └──────────────┼──────────────┘
                            │
                       Pull Request
                            │
                            ▼
                           dev
```

**Thank you for contributing to ModelBrief!** 🚀
