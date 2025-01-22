
import pylab as py
import numpy as np
import itertools as it
import argparse
import json

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="SVM tuning")

parser.add_argument('-experiment', default=1, dest='experiment', type=int)
parser.add_argument('-n_features', default=18, dest='n_features', type=int)
parser.add_argument('-params', default='svm_linear_grid.json', dest='params', type=str)

args = parser.parse_args()

def test_svm(x_train, y_train, x_test, y_test, args):
    # Fit the model
    svc = SVC(**args)
    svc.fit(x_train, y_train)

    # Predict
    y_train_pred = svc.predict(x_train)
    y_test_pred = svc.predict(x_test)

    return accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)

def load_and_preprocess_data(experiment, selected_features):
    # Load data
    x_train = np.loadtxt("../../data/x_train_" + str(experiment) + ".csv")
    y_train = np.loadtxt("../../data/y_train_" + str(experiment) + ".csv", dtype=np.int32)
    x_validation = np.loadtxt("../../data/x_validation_" + str(experiment) + ".csv")
    y_validation = np.loadtxt("../../data/y_validation_" + str(experiment) + ".csv", dtype=np.int32)

    # Choose features
    x_train = x_train[:, selected_features]
    x_validation = x_validation[:, selected_features]

    # Standardize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)

    return x_train, y_train, x_validation, y_validation

def grid_search(x_train, y_train, x_test, y_test, parameters):
    best_score = 0
    best_args = None

    for args in it.product(*parameters.values()):
        args = dict(zip(parameters.keys(), args))
        train_score, test_score = test_svm(x_train, y_train, x_test, y_test, args)
        print(args, train_score, test_score)
        if test_score > best_score:
            best_score = test_score
            best_args = args

    return best_score, best_args

if __name__ == "__main__":
    # Choose features
    selected_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    selected_features = selected_features[:args.n_features]

    # Load data
    x_train, y_train, x_validation, y_validation = load_and_preprocess_data(args.experiment, selected_features)

    # Load parameters
    with open("../../configuration/" + args.params) as f:
        parameters = json.load(f)

    # Grid search
    best_score, best_args = grid_search(x_train, y_train, x_validation, y_validation, parameters)

    # Print results
    print("Best configuration:", best_args)
    print("Best score:", best_score)
