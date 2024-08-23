from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        # calculating s_i
        s = -self.loss_derivative(y, predictions)

        # generating bootstrapped train sample
        idx_bootstrapped = np.random.choice(x.shape[0], size=int(x.shape[0] * self.subsample), replace=True)
        x_bootstrapped = x[idx_bootstrapped]
        s_bootstrapped = s[idx_bootstrapped]

        # creating and fitting new base model
        new_base_model = self.base_model_class(**self.base_model_params)
        new_base_model.fit(x_bootstrapped, s_bootstrapped)

        # finding optimal gamma
        new_opt_gamma = self.find_optimal_gamma(y, predictions, new_base_model.predict(x))

        # adding results
        self.gammas.append(new_opt_gamma)
        self.models.append(new_base_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        # zero model
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        # adding info to history
        self.history["loss_train"] = [self.loss_fn(y_train, train_predictions)]
        self.history["loss_val"] = [self.loss_fn(y_valid, valid_predictions)]

        for _ in range(self.n_estimators):
            # fitting new base model
            self.fit_new_base_model(x_train, y_train, train_predictions)

            # updating predictions on train and val
            train_predictions += (self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train))
            valid_predictions += (self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid))

            # adding info to history
            self.history["loss_train"] += [self.loss_fn(y_train, train_predictions)]
            self.history["loss_val"] += [self.loss_fn(y_valid, valid_predictions)]

            # checking if we need to stop fitting
            if self.early_stopping_rounds is not None:
                if self.history["loss_val"][-2] <= self.history["loss_val"][-1]:
                    self.curr_rounds_till_stop -= 1
                else:
                    self.curr_rounds_till_stop = self.early_stopping_rounds
                if self.curr_rounds_till_stop == 0:
                    break

        # drawing a plot if needed
        if self.plot:
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            estimators = np.arange(self.n_estimators + 1)

            # first plot
            ax[0].plot(estimators, self.history["loss_train"])
            ax[0].set_title("Loss on Train by Number of Base Estimators")
            ax[0].set_xlabel("Base Estimators")
            ax[0].set_ylabel("Loss")

            # second plot
            ax[1].plot(estimators, self.history["loss_val"])
            ax[1].set_title("Loss on Val by Number of Base Estimators")
            ax[1].set_xlabel("Base Estimators")
            ax[1].set_ylabel("Loss")

            fig.tight_layout(h_pad=2)
            plt.show()

    def predict_proba(self, x):
        # calculating aggregated predictions
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += (self.learning_rate * gamma * model.predict(x))
        # calculating probabilities from predictions
        proba1 = self.sigmoid(predictions)
        proba0 = 1. - proba1

        proba = np.concatenate([proba0.reshape((-1, 1)), proba1.reshape((-1, 1))], axis=1)
        return proba

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
