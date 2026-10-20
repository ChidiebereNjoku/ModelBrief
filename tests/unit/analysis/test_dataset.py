import numpy as np

from modelbrief.analysis.dataset import describe_dataset


def test_missing_data_split():
    assert describe_dataset(None) == {"available": False}


def test_numpy_array_summary():
    info = describe_dataset(np.zeros((5, 3)))

    assert info["available"] is True
    assert info["rows"] == 5
    assert info["shape"] == (5, 3)
    assert info["type"] == "ndarray"
    assert info["missing_values"] == 0


def test_missing_values_are_counted():
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [5.0, 6.0]])

    assert describe_dataset(X)["missing_values"] == 2


def test_missing_values_not_checked_for_non_float_data():
    assert describe_dataset(np.array([[1, 2], [3, 4]]))["missing_values"] is None


def test_plain_list_uses_its_length():
    info = describe_dataset([[1, 2], [3, 4], [5, 6]])

    assert info["rows"] == 3
    assert info["type"] == "list"


def test_target_summary():
    info = describe_dataset(np.zeros((4, 2)), y=[0, 1, 1, 2])

    assert info["target_shape"] == (4,)
    assert info["target_unique"] == 3
