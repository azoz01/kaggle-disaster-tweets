from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import numpy as np


def plot_learning_curve(estimator: object,
                        X: np.ndarray,
                        y: np.ndarray) -> plt.figure:
    """
    Returns figure with learning curve for train and validation set.
    Lines are mean accuracy, shaded area is mean +/- standard
    deviation of accuray and points are min/max accuracy

    Args:
        estimator (object): type that implements the 'fit'
            and 'predict' methods
        X (np.ndarray): feature variables
        y (np.ndarray): target variable

    Returns:
        plt.figure:
    """

    lc = learning_curve(estimator, X, y,
                        train_sizes=np.linspace(.1, 1, 10),
                        random_state=42,
                        scoring=make_scorer(accuracy_score))

    sizes = lc[0]
    train_mean = lc[1].mean(axis=1)
    train_std = lc[1].std(axis=1)
    valid_mean = lc[2].mean(axis=1)
    valid_std = lc[2].std(axis=1)

    fig = plt.figure()
    plt.plot(sizes, train_mean, label='train score')
    plt.fill_between(sizes, train_mean - train_std,
                     train_mean + train_std, alpha=.1)
    plt.scatter(sizes, lc[1].min(axis=1), color='#1f77b4')
    plt.scatter(sizes, lc[1].max(axis=1), color='#1f77b4')
    plt.plot(sizes, valid_mean, label='valid score')
    plt.fill_between(sizes, valid_mean - valid_std,
                     valid_mean + valid_std, alpha=.1)
    plt.scatter(sizes, lc[2].min(axis=1), color='#ff7f0e')
    plt.scatter(sizes, lc[2].max(axis=1), color='#ff7f0e')
    plt.plot([sizes[0], sizes[-1]], [.8, .8])
    plt.suptitle('Accuracy of model')
    plt.xlabel('Train dataset size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.rcParams.update({
        'figure.facecolor': (1, 1, 1, 1)
    })

    return fig
