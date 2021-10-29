import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import pandas as pd
import numpy as np
import cv2
import h5py
import os

import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleClassifier, GradientClassificationConfidence
from keras_uncertainty.utils import numpy_entropy, classifier_calibration_error

def class_sampling(X, y, samplesPerClass, numberOfClasses):
    X_ret = np.zeros((samplesPerClass * numberOfClasses, X.shape[1], X.shape[2], X.shape[3]), dtype = np.float32)
    y_ret = np.zeros((samplesPerClass * numberOfClasses, ), dtype = np.uint8)
    count = 0

    for classIdx in range(numberOfClasses):
        indices = np.where(y == classIdx)[0]
        doResample = len(indices) < samplesPerClass

        chosenIndices = np.random.choice(indices, samplesPerClass, replace = doResample)

        for ci in chosenIndices:
            X_ret[count] = X[ci]
            y_ret[count] = y[ci]

            count += 1

    return X_ret, y_ret

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    transform_fn = lambda x: cv2.copyMakeBorder(x, 2, 2, 2, 2, cv2.BORDER_REFLECT)
    x_train = np.array(list(map(transform_fn, x_train)))
    x_test = np.array(list(map(transform_fn, x_test)))

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = x_train.reshape((-1, 32, 32, 1))
    x_test = x_test.reshape((-1, 32, 32, 1))

    return x_train, y_train, x_test, y_test

def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    transform_fn = lambda x: cv2.copyMakeBorder(x, 2, 2, 2, 2, cv2.BORDER_REFLECT)
    x_train = np.array(list(map(transform_fn, x_train)))
    x_test = np.array(list(map(transform_fn, x_test)))

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = x_train.reshape((-1, 32, 32, 1))
    x_test = x_test.reshape((-1, 32, 32, 1))

    return x_train, y_train, x_test, y_test

def load_svhn():
    data = h5py.File('svhn_32x32.hdf5', 'r')

    x_train = data['x_train'][...]
    y_train = data['y_train'][...]

    x_test = data['x_test'][...]
    y_test = data['y_test'][...]

    #Normalization
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test

CONFIGS = {
    "mnist": {
        "num_classes": 10,
        "loader": load_mnist,
    },
    "fashion_mnist": {
        "epochs": 100,
        "batch_size": 64,
        "num_classes": 10,
        "input_shape": (32, 32, 1),
        "loader": load_fashion_mnist,
        "ood_dataset": "mnist"
    },
    "cifar10": {
        "epochs": 100,
        "batch_size": 64,
        "num_classes": 10,
        "input_shape": (32, 32, 3),
        "loader": load_cifar10,
        "ood_dataset": "svhn"
    },
    "svhn": {
        "num_classes": 10,
        "loader": load_svhn,
    }
}

from uncertainty_methods_miniVGG import minivgg_baseline as unc_baseline

def upper_whisker(data):
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    hival = Q3 + 1.5 * IQR

    wiskhi = np.compress(data <= hival, data)

    if len(wiskhi) == 0:
        actual_hival = np.max(data)
    else:
        actual_hival = np.max(wiskhi)

    return actual_hival

def lower_whisker(data):
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    loval = Q3 - 1.5 * IQR

    wisklo = np.compress(data >= loval, data)

    if len(wisklo) == 0:
        actual_loval = np.min(data)
    else:
        actual_loval = np.min(wisklo)

    return actual_loval

NUM_ENSEMBLES = 5
SAMPLES_PER_CLASS = [1, 5, 10, 50, 100, 250, 500, 1000, 5000]
TRIALS = 1

import argparse

def predict_probs(grad_model, data):
    confidences = grad_model.predict(data)
    confidences = (confidences - min(confidences)) / (max(confidences) - min(confidences))
    maxprob = 1.0 - confidences

    return maxprob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--agg_metric", help="Name of the aggregation metric to use (l1_norm/l2_norm/min/max/mean/std)", default="l2_norm")

    args = parser.parse_args()
    dataset = args.dataset
    agg_metric = args.agg_metric

    uncert_method = "gradient-{}".format(agg_metric)
    model_name = "miniVGG"

    ds_info = CONFIGS[dataset]
    num_classes = ds_info["num_classes"]
    input_shape = ds_info["input_shape"]
    batch_size = ds_info["batch_size"]
    epochs = ds_info["epochs"] 
    output_csv_filename = 'maxprob-vs-SPC-{}-results-{}-{}.csv'.format(dataset, model_name, uncert_method)

    x_train, y_train, x_test, y_test = ds_info["loader"]()
    y_test = to_categorical(y_test, num_classes)

    ood_ds_info = CONFIGS[ds_info["ood_dataset"]]
    _, __, ood_x_test, ___ = ood_ds_info["loader"]()

    x_ood = np.concatenate([x_test, ood_x_test])
    y_ood = np.concatenate([np.zeros((x_test.shape[0],)), np.ones((ood_x_test.shape[0],))])

    print("X_train shape {} y_train shape {}".format(x_train.shape, y_train.shape))
    print("X_test shape {} y_test shape {}".format(x_test.shape, y_test.shape))

    results = pd.DataFrame(columns = ['spc', 'acc', 'mean_maxprob', 'std_maxprob', 'median_maxprob', 'uq_maxprob', 'lq_maxprob', 'uw_maxprob', 'lw_maxprob',
                                                    'ece', 'ood_auc_maxprob', 'brier', 'train_ece', 'tr_test_auc_maxprob', 'tr_ood_auc_maxprob'])

    for spc in SAMPLES_PER_CLASS:
        accuracies = []

        for i in range(TRIALS):
            x_sample, y_sample = class_sampling(x_train, y_train, spc, num_classes)
            y_sample = to_categorical(y_sample, num_classes)

            # Possibly we need data augmentation here
            model = unc_baseline(input_shape, (x_sample, y_sample), epochs, batch_size)
            grad_model = GradientClassificationConfidence(model, aggregation=args.agg_metric)

            preds = model.predict(x_test, verbose=0, batch_size=128)            
            maxprob = predict_probs(grad_model, x_test)
            
            #loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

            class_true = np.argmax(y_test, axis=1)
            class_pred = np.argmax(preds, axis=1)

            # Standard metrics
            acc = accuracy_score(class_true, class_pred)
            err = 100.0 * (1.0 - acc)

            # Max probability box plots
            mean_maxprob = np.mean(maxprob)
            std_maxprob = np.std(maxprob)

            median_maxprob = np.median(maxprob)
            uq_maxprob = np.quantile(maxprob, q=0.75)
            lq_maxprob = np.quantile(maxprob, q=0.25)
            
            uw_maxprob = upper_whisker(maxprob)
            lw_maxprob = lower_whisker(maxprob)

            brier = mean_squared_error(np.max(y_test, axis=-1), maxprob)
            brier = round(brier, 3)

            # Calibration error
            class_conf = maxprob
            cab_err = classifier_calibration_error(class_pred, class_true, class_conf, weighted=True)
            cab_err = round(cab_err, 3)

            train_preds = model.predict(x_sample, verbose=0, batch_size=128)
            train_class_conf = predict_probs(grad_model, x_sample)

            train_class_preds = np.argmax(train_preds, axis=1)
            train_class_true = np.argmax(y_sample, axis=1)
            train_cab_err = classifier_calibration_error(train_class_preds, train_class_true, train_class_conf, weighted=True)

            #OOD Performance  
            ood_preds = model.predict(x_ood)
            ood_score_maxprob = predict_probs(grad_model, x_ood)            
            auc_maxprob = roc_auc_score(y_ood, 1.0 - ood_score_maxprob)

            train_maxprob = 1.0 - train_class_conf
            test_maxprob = ood_score_maxprob[:x_test.shape[0]]
            tr_test_maxprob = np.concatenate([train_maxprob, 1.0 -  test_maxprob])

            tr_test_ood_y = np.concatenate([np.zeros(train_preds.shape[0]), np.ones(x_test.shape[0])])
            tr_test_auc_maxprob = roc_auc_score(tr_test_ood_y, 1.0 - tr_test_maxprob)

            tr_ood_maxprob = np.concatenate([train_maxprob, ood_score_maxprob[x_test.shape[0]:]])
            tr_ood_y = np.concatenate([np.zeros(train_maxprob.shape[0]), np.ones(ood_score_maxprob[x_test.shape[0]:].shape[0])])
            tr_ood_auc_maxprob = roc_auc_score(tr_ood_y, tr_ood_maxprob)

            acc = round(100 * acc, 3)

            mean_maxprob = round(mean_maxprob, 3)
            median_maxprob = round(median_maxprob, 3)
            std_maxprob = round(std_maxprob, 3)

            auc_maxprob = round(auc_maxprob, 3)

            print("SPC {} Test Accuracy: {:.3f} Maxprob {:.3f} +- {:.3f} ECE {:.3f} AUC MP {:.3f}".format(spc, acc, mean_maxprob, std_maxprob, cab_err, auc_maxprob))
            
            results_dict = {'spc': spc, 'acc': acc, 'mean_maxprob': mean_maxprob, 'std_maxprob': std_maxprob, 'ece': cab_err, 'ood_auc_maxprob': auc_maxprob, 'brier': brier}

            results_dict['median_maxprob'] = median_maxprob
            results_dict['uq_maxprob'] = uq_maxprob
            results_dict['lq_maxprob'] = lq_maxprob
            results_dict['uw_maxprob'] = uw_maxprob
            results_dict['lw_maxprob'] = lw_maxprob

            results_dict['train_ece'] = train_cab_err

            results_dict['tr_test_auc_maxprob'] = tr_test_auc_maxprob
            results_dict['tr_ood_auc_maxprob'] = tr_ood_auc_maxprob

            for k, v in results_dict.items():
                results_dict[k] = np.round(v, 3)

            results = results.append(results_dict,  ignore_index=True)
            accuracies.append(acc)

            results.to_csv(output_csv_filename, sep=';', index=False)

            del model

        # Possible do the same for entropy, or compute some histogram or distribution metrics.
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        mean_acc = round(mean_acc, 3)
        std_acc = round(std_acc, 3)

        print("After {} trials - Accuracy is {} +- {}".format(TRIALS, mean_acc, std_acc))
    
    results.to_csv(output_csv_filename, sep=';', index=False)