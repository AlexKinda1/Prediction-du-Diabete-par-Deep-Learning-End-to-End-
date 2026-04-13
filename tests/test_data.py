import numpy as np

def test_data_no_nan():
    data = np.array([1.0, 2.0, 3.0])
    assert not np.isnan(data).any()


def test_shape():
    data = np.zeros((3, 224, 224, 3))
    assert data.shape == (3, 224, 224, 3)
