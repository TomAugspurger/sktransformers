import numpy as np
import pandas as pd
from sktransformers.feature_selection import NaNVarianceThreshold
import pandas.util.testing as tm
import pytest

@pytest.fixture
def data():
    return np.arange(100).reshape(10, 10)


class TestNaNVarianceThreshold:

    def test_numpy(self, data):
        trn = NaNVarianceThreshold()
        X_ = trn.fit_transform(data)
        np.testing.assert_allclose(data, X_)

    def test_numpympy_drop(self, data):
        data[:, 0] = 0
        trn = NaNVarianceThreshold()
        X_ = trn.fit_transform(data)
        expected = data[:, 1:]
        np.testing.assert_allclose(expected, X_)

    def test_pandas(self, data):
        trn = NaNVarianceThreshold()
        df = pd.DataFrame(data)
        X_ = trn.fit_transform(df)
        tm.assert_frame_equal(X_, df)

    def test_pandas_drop(self, data):
        data[:, 0] = 0
        trn = NaNVarianceThreshold()
        df = pd.DataFrame(data)
        X_ = trn.fit_transform(df)
        expected = df.iloc[:, 1:]
        tm.assert_frame_equal(X_, expected)
