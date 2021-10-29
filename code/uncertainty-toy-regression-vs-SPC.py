import numpy as np
import math

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.utils import to_categorical

import keras_uncertainty
from keras_uncertainty.models import MCDropoutRegressor, DeepEnsembleRegressor, StochasticRegressor, TwoHeadStochasticRegressor
from keras_uncertainty.layers import DropConnectDense, BayesByBackpropDense, FlipoutDense
from keras_uncertainty.losses import regression_gaussian_nll_loss

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

def toy_function(input):
    output = []

    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)

        out = math.sin(inp) + np.random.normal(0, std)
        output.append(10 * out)

    return np.array(output)

def train_standard_model(x_train, y_train, domain):
    inp = Input(shape=(1,))
    x = Dense(32, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    mean = Dense(1, activation="linear")(x)
    var = Dense(1, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")
    train_model.fit(x_train, y_train, verbose=2, epochs=500)

    mean_pred, var_pred = pred_model.predict(domain)
    std_pred = np.sqrt(var_pred)

    return mean_pred, std_pred

def train_dropout_model(x_train, y_train, domain, prob=0.2):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(1,)))
    model.add(Dropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(prob))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(x_train, y_train, verbose=2, epochs=500)

    mc_model = MCDropoutRegressor(model)
    pred_mean, pred_std = mc_model.predict(domain, num_samples=NUM_SAMPLES, batch_size=1)

    return pred_mean, pred_std

def train_dropconnect_model(x_train, y_train, domain, prob=0.05, drop_noise_shape=None):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(1,), prob=prob, drop_noise_shape=drop_noise_shape))
    model.add(DropConnectDense(32, activation="relu", prob=prob, drop_noise_shape=drop_noise_shape))
    model.add(DropConnectDense(1, activation="linear", drop_noise_shape=drop_noise_shape))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(x_train, y_train, verbose=2, epochs=500)

    mc_model = MCDropoutRegressor(model)
    pred_mean, pred_std = mc_model.predict(domain, num_samples=NUM_SAMPLES, batch_size=1)

    return pred_mean, pred_std

def train_ensemble_model(x_train, y_train, domain):
    def model_fn():
        inp = Input(shape=(1,))
        x = Dense(32, activation="relu")(inp)
        x = Dense(32, activation="relu")(x)
        mean = Dense(1, activation="linear")(x)
        var = Dense(1, activation="softplus")(x)

        train_model = Model(inp, mean)
        pred_model = Model(inp, [mean, var])

        train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")

        return train_model, pred_model
    
    model = DeepEnsembleRegressor(model_fn, num_estimators=10)
    model.fit(x_train, y_train, verbose=2, epochs=500)
    pred_mean, pred_std = model.predict(domain)

    return pred_mean, pred_std

def train_flipout_model(x_train, y_train, domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches

    model = Sequential()
    model.add(FlipoutDense(32, kl_weight, prior=False, bias_distribution=True, activation="relu", input_shape=(1,)))
    model.add(FlipoutDense(32, kl_weight, prior=False, bias_distribution=True, activation="relu"))
    model.add(FlipoutDense(1, kl_weight, prior=False, bias_distribution=True, activation="linear"))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(x_train, y_train, verbose=2, epochs=700)
    
    st_model = StochasticRegressor(model)
    pred_mean, pred_std = st_model.predict(domain, num_samples=NUM_SAMPLES)

    return pred_mean, pred_std

def train_flipout_nll_model(x_train, y_train, domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches

    inp = Input(shape=(1,))
    x = FlipoutDense(32, kl_weight, prior=False, bias_distribution=True, activation="relu",)(inp)
    x = FlipoutDense(32, kl_weight, prior=False, bias_distribution=True, activation="relu")(x)
    mean = FlipoutDense(1, kl_weight, prior=False, bias_distribution=True, activation="linear")(x)
    var = FlipoutDense(1, kl_weight, prior=False, bias_distribution=True, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")
    train_model.fit(x_train, y_train, verbose=2, epochs=700)

    st_model = TwoHeadStochasticRegressor(pred_model)
    pred_mean, std_pred = st_model.predict(domain, num_samples=NUM_SAMPLES)

    return pred_mean, std_pred

METHODS = {
    "Classical NN": train_standard_model,
    "Dropout": train_dropout_model,
    "DropConnect": train_dropconnect_model,
    "5 Ensembles": train_ensemble_model,
    "Flipout": train_flipout_model,
    "Flipout + NLL": train_flipout_nll_model,
    "Training\nSamples": None,
}

NUM_SAMPLES = 30
TRAIN_SAMPLES = [25, 50, 100, 200]

if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=len(METHODS.keys()), ncols=len(TRAIN_SAMPLES), figsize=(6, 10))
    methods = list(METHODS.keys())

    x_test = np.linspace(-7.0, 7.0, num=200)    
    y_test = toy_function(x_test)

    domain = np.linspace(-7.0, 7.0, num=400)
    domain = domain.reshape((-1, 1))

    for i, num_samples in enumerate(TRAIN_SAMPLES):
        x_train = np.linspace(-4.0, 4.0, num=num_samples)
        y_train = toy_function(x_train)

        for j, method_name in enumerate(methods):
            ax = axes[j][i]

            if METHODS[method_name] is None:
                ax.scatter(x_train, y_train, s=8, marker='o', color="black")
                ax.set_xlim([-7.0, 7.0])
            else:
                y_pred_mean, y_pred_std = METHODS[method_name](x_train, y_train, domain)

                y_pred_mean = y_pred_mean.reshape((-1,))
                y_pred_std = y_pred_std.reshape((-1,))
                y_pred_up_1 = y_pred_mean + y_pred_std
                y_pred_down_1 = y_pred_mean - y_pred_std

                ax.fill_between(domain.ravel(), y_pred_down_1, y_pred_up_1, color=(0, 0, 0.9, 0.7), label="One Sigma Confidence Interval")
                ax.plot(domain.ravel(), y_pred_mean, '.', color=(0, 0.9, 0.0, 0.8), markersize=0.2, label="Mean")

            ax.set_ylim([-20.0, 20.0])

            ax.axvline(x=-4.0, color="black")
            ax.axvline(x= 4.0, color="black")

            if i >= 0 and j == 0:
                ax.set_title("{} Samples".format(num_samples))

            if j >= 0 and i == 0:
                ax.set_ylabel(method_name)

            if not (j >= 0 and i == len(TRAIN_SAMPLES) - 1):
                ax.get_yaxis().set_ticks([])

            if not (i >= 0 and j == len(METHODS) - 1):
                ax.get_xaxis().set_ticks([])

            ax.yaxis.tick_right()
            
            for item in ax.get_xticklabels():
                item.set_fontsize(7)

            for item in ax.get_yticklabels():
                item.set_fontsize(7)

    plt.savefig("uncertainty-toy-regression-vs-SPC.pdf", bbox_inches="tight")
    plt.show()