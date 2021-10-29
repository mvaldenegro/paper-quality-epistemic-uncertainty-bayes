import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

import keras_uncertainty
from keras_uncertainty.models import MCDropoutClassifier, DeepEnsembleClassifier, StochasticClassifier, GradientClassificationConfidence
from keras_uncertainty.layers import duq_training_loop, add_gradient_penalty, add_l2_regularization
from keras_uncertainty.layers import DropConnectDense, BayesByBackpropDense, RBFClassifier, FlipoutDense
from keras_uncertainty.utils import numpy_entropy

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True
})

def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)

def train_standard_model(x_train, y_train, domain):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=50)

    preds = model.predict(domain)
    entropy = uncertainty(preds)

    return entropy

def train_dropout_model(x_train, y_train, domain, prob=0.25):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(prob))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=50)

    mc_model = MCDropoutClassifier(model)
    preds = mc_model.predict(domain, num_samples=NUM_SAMPLES)
    entropy = uncertainty(preds)

    return entropy

def train_dropconnect_model(x_train, y_train, domain, prob=0.25):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(2,), prob=prob))
    model.add(DropConnectDense(32, activation="relu", prob=prob))
    model.add(DropConnectDense(2, activation="softmax", prob=prob))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=50)

    mc_model = MCDropoutClassifier(model)
    preds = mc_model.predict(domain, num_samples=NUM_SAMPLES)
    entropy = uncertainty(preds)

    return entropy

def train_ensemble_model(x_train, y_train, domain):
    def model_fn():
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(2,)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    model = DeepEnsembleClassifier(model_fn, num_estimators=5)
    model.fit(x_train, y_train, verbose=2, epochs=50)
    preds = model.predict(domain)
    entropy = uncertainty(preds)

    return entropy

def train_flipout_model(x_train, y_train, domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches
    
    model = Sequential()
    model.add(FlipoutDense(32, kl_weight, prior=False, bias_distribution=False, activation="relu", input_shape=(2,)))
    model.add(FlipoutDense(32, kl_weight, prior=False, bias_distribution=False, activation="relu"))
    model.add(FlipoutDense(2, kl_weight, prior=False, bias_distribution=False, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy", "sparse_categorical_crossentropy"])

    model.fit(x_train, y_train, verbose=2, epochs=300)
    st_model = StochasticClassifier(model)

    preds = st_model.predict(domain, num_samples=NUM_SAMPLES)
    entropy = uncertainty(preds)

    return entropy

def train_duq_model(x_train, y_train, domain):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dense(32, activation="relu"))
    model.add(RBFClassifier(2, 0.1, centroid_dims=2, trainable_centroids=True))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    add_gradient_penalty(model, lambda_coeff=0.5)
    add_l2_regularization(model)

    y_train = to_categorical(y_train, num_classes=2)

    model.fit(x_train, y_train, verbose=2, epochs=50)

    preds = model.predict(domain)
    confidence = np.max(preds, axis=1)

    return confidence

def train_gradient_model(x_train, y_train, domain):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    y_train = to_categorical(y_train, num_classes=2)
    model.fit(x_train, y_train, verbose=2, epochs=50)

    grad_model = GradientClassificationConfidence(model, aggregation="l1_norm")
    conf = grad_model.predict(domain)

    return np.array(conf)

METHODS = {
    "Baseline": train_standard_model,
    "Dropout": train_dropout_model,
    "DropConnect": train_dropconnect_model,
    "5 Ensembles": train_ensemble_model,
    "Flipout": train_flipout_model,
    "DUQ": train_duq_model,
    "Gradient L1": train_gradient_model
}

NUM_SAMPLES = 30
TRAIN_SAMPLES = [10, 25, 50, 100]

if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=len(METHODS.keys()), ncols=len(TRAIN_SAMPLES), figsize=(6, 10))
    methods = list(METHODS.keys())
    
    min_x, max_x = [-2, -2] , [3, 2]
    res = 0.05

    xx, yy = np.meshgrid(np.arange(min_x[0], max_x[0], res), np.arange(min_x[1], max_x[1], res))
    domain = np.c_[xx.ravel(), yy.ravel()]

    import matplotlib.pylab as pl
    from matplotlib.colors import ListedColormap
    cmap = pl.cm.binary
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = 0.7
    my_cmap = ListedColormap(my_cmap)

    for i, num_samples in enumerate(TRAIN_SAMPLES):
        x, y = make_moons(n_samples=num_samples, noise=0.1, random_state=749)

        for j, method_name in enumerate(methods):
            ax = axes[j][i]

            domain_conf = METHODS[method_name](x, y, domain)
            domain_conf = domain_conf.reshape(xx.shape)

            ax.contourf(xx, yy, domain_conf)
            ax.scatter(x[:, 0], x[:, 1], c=y, cmap=my_cmap)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if i >= 0 and j == 0:
                ax.set_title("{} Samples".format(num_samples))

            if j >= 0 and i == 0:
                ax.set_ylabel(method_name)

    plt.savefig("uncertainty-two-moons-vs-SPC.pdf", bbox_inches="tight")
    plt.show()