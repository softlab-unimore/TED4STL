import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, ParameterGrid


def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]

    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]

    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr

def fit_mlp(train_features,  train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]

    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(50,), (100,), (50,50)]
    }

    valid_results = []
    for params in ParameterGrid(param_grid):
        mlp = MLPRegressor(max_iter=1000, early_stopping=True, **params)
        mlp.fit(train_features, train_y)
        valid_pred = mlp.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)

    best_params = list(ParameterGrid(param_grid))[np.argmin(valid_results)]
    mlp = MLPRegressor(max_iter=1000, early_stopping=True, **best_params)
    mlp.fit(train_features, train_y)
    return mlp