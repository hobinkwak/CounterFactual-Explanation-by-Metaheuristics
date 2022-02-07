import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def get_dataset(n_samples=100000, n_features=10, n_informative=7, n_targets=5, random_state=42):
    X, y, coef = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                 n_targets=n_targets, random_state=random_state, coef=True)

    not_informative_idx = np.unique(np.where(coef == 0)[0])

    return X, y, coef, not_informative_idx

def split_dataset(X, y, train_size=0.7, test_size=0.15, train_shuffle=True, test_shuffle=True, random_state=42, save=True):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=(1-train_size), shuffle=train_shuffle, random_state=random_state)
    valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size=test_size, shuffle=test_shuffle, random_state=random_state)
    if save:
        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)
        np.save('valid_x.npy', valid_x)
        np.save('valid_y.npy', valid_y)
        np.save('test_x.npy', test_x)
        np.save('test_y.npy', test_y)
    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

if __name__ == '__main__':
    X, y, coef, nidx = get_dataset(n_samples=200000, n_features=10, n_informative=7, n_targets=3)
    split_dataset(X, y)

