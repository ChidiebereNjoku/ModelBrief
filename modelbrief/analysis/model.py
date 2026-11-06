def is_fitted(model):
    """Determine whether a supported estimator has been fitted.

    Parameters
    ----------
    model : object
        Model instance to inspect. The fitted state is checked only for
        scikit-learn-style estimators that provide ``fit`` and ``get_params``.

    Returns
    -------
    bool or None
        ``True`` if the estimator is fitted, ``False`` if it is not fitted,
        or ``None`` if its fitted state cannot be determined. Deep-learning
        models and custom objects without a supported fitted-state signal have
        no reliable signal available to this check and are reported as unknown,
        rather than being classified by guesswork.

    Notes
    -----
    Fitted-state detection uses
    :func:`sklearn.utils.validation.check_is_fitted`, which supports
    scikit-learn estimators, including pipelines. If scikit-learn is unavailable,
    the result is ``None``.
    """
    if not (hasattr(model, "fit") and hasattr(model, "get_params")):
        return None
    try:
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted
    except ImportError:
        return None
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    except TypeError:  # For Non Sklearn Models: Can be set to generic response
        return None


def model_overview(model, adapter, task):
    """Build a summary of a model for inclusion in a report.

    Parameters
    ----------
    model : object
        Model instance to describe.
    adapter : object
        Model adapter providing a ``parameters()`` method for model-specific
        parameter information.
    task : str
        Task associated with the model, such as classification or regression.

    Returns
    -------
    dict
        Model overview with the class name (``model_class``), defining module
        (``module``), task (``task``), fitted-state result
        (``fitted_model_supplied``), a flag indicating that no retraining was
        performed (``retraining_performed``), and adapter-provided parameter
        details (``parameters``).

    Notes
    -----
    This function describes the supplied model; it does not fit or retrain it.
    """
    return {
        "model_class": type(model).__name__,
        "module": type(model).__module__,
        "task": task,
        "fitted_model_supplied": is_fitted(model),
        "retraining_performed": False,
        "parameters": adapter.parameters(),
    }