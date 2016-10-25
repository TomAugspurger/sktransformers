import numpy as np
import pandas as pd
import dask.dataframe as dd

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectFromModel
from sktransformers.preprocessing import CategoricalEncoder, DummyEncoder, Imputer


N_NUM = 100
N_CAT = 10
N_ROW = 10000


def generate():
    np.random.seed(2)
    numeric = pd.DataFrame(np.random.uniform(size=(N_ROW, N_NUM)),
                           columns=[str(x) for x in range(N_NUM)])
    categorical = pd.DataFrame(
        np.random.choice(list('abcdefg'), size=(N_ROW, N_CAT)),
        columns=[str(x) for x in range(N_NUM, N_NUM + N_CAT)])

    df = pd.concat([numeric, categorical], axis=1)
    valid = df.where(np.random.uniform(size=df.shape) < .95)

    coef = pd.Series(np.zeros_like(numeric.columns), index=numeric.columns)
    coef[:10] = np.random.uniform(low=1, high=10, size=10)
    y = (numeric * coef).sum(1)
    y += 20 * (categorical.iloc[:, 0] == 'a')

    return valid, y


def fit():
    X, y = generate()
    dX = dd.from_pandas(X, npartitions=10)
    y = dd.from_pandas(y, npartitions=10)

    pipe = make_pipeline(
        CategoricalEncoder(),
        DummyEncoder(),
        Imputer(),
    )

    X_ = pipe.fit_transform(dX)
    clf = SGDRegressor()

    for i in range(X_.npartitions):
        for j in range(5):
            print(i, j)
            X_sub = X_.get_partition(i).compute()
            y_sub = y.get_partition(i).compute()
            clf.partial_fit(X_sub, y_sub)

    sfm = SelectFromModel(clf, prefit=True)
    return pipe, clf, sfm

