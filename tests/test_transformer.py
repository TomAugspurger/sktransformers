import pytest

import pandas as pd
import pandas.util.testing as tm
from sktransformers.preprocessing import CategoricalEncoder, DummyEncoder


@pytest.fixture(scope="module")
def raw():
    d = {"A": ['a', 'b', 'c', 'a'],
         "B": ['a', 'b', 'c', 'a'],
         "C": ['a', 'b', 'c', 'a'],
         "D": [1, 2, 3, 4],
         }

    return pd.DataFrame(d, columns=sorted(d.keys()))


@pytest.fixture(scope='module')
def data():
    df = pd.DataFrame(
        {"A": pd.Categorical(['a', 'b', 'c', 'a'], ordered=True),
         "B": pd.Categorical(['a', 'b', 'c', 'a'], ordered=False),
         "C": pd.Categorical(['a', 'b', 'c', 'a'],
                             categories=['a', 'b', 'c', 'd']),
         "D": [1, 2, 3, 4],
         }
    )
    return df


class TestCategoricalEncoder:

    def test_ce(self, raw):
        ce = CategoricalEncoder()
        trn = ce.fit_transform(raw)
        assert trn['A'].dtype == 'category'
        assert trn['B'].dtype == 'category'
        assert trn['C'].dtype == 'category'
        assert trn['D'].dtype == int
        assert all(ce.cat_cols_ == pd.Index(["A", "B", "C"]))

    def test_categories(self, raw):
        cats = ['a', 'b', 'c', 'd']
        ce = CategoricalEncoder(categories={'A': cats},
                                ordered={'A': True})
        trn = ce.fit_transform(raw)
        assert trn['A'].dtype == 'category'
        assert all(trn['A'].cat.categories == cats)
        assert trn['A'].cat.ordered


class TestDummyEncoder:

    def test_smoke(self, data):
        ct = DummyEncoder()
        ct = ct.fit(data)
        trn = ct.transform(data)
        assert trn.shape[1] == 11

    def test_drop_first(self, data):
        ct = DummyEncoder(drop_first=True)
        trn = ct.fit_transform(data)
        assert trn.shape[1] == 8
