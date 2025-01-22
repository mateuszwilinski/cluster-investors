
import pylab as py
import numpy as np
import scipy.stats as stats
import argparse

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

py.rcParams["font.family"] = "serif"
py.rcParams["mathtext.fontset"] = "cm"
py.rcParams['pdf.fonttype'] = 42
py.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser(description="SVM classification")

parser.add_argument('-experiment', default=1, dest='experiment', type=int)
parser.add_argument('-n_features', default=18, dest='n_features', type=int)
parser.add_argument('-epochs', default=20, dest='epochs', type=int)
parser.add_argument('-batch', default=32, dest='batch_size', type=int)
parser.add_argument('-l1', default=256, dest='l1', type=int)
parser.add_argument('-l2', default=1024, dest='l2', type=int)
parser.add_argument('-l3', default=1024, dest='l3', type=int)
parser.add_argument('-drop', default=0.2, dest='dropout', type=float)
parser.add_argument('-lr', default=1e-3, dest='lr', type=float)
parser.add_argument('-plot_confusion', default=True, dest='plot_confusion', type=bool)
parser.add_argument('-print_report', default=True, dest='print_report', type=bool)
parser.add_argument('-plot_explanation', default=True, dest='plot_explanation', type=bool)

args = parser.parse_args()

features = ["buy ratio", "market orders ratio", "mean order size",
            "std of order size", "mean order creation times", "std order creation times",
            "cancellation ratio", "no trades", "traded volume",
            "short_trend", "short_dir_trend", "trend", "dir_trend", "long_trend", "long_dir_trend",
            "profit", "long_profit", "weight_profit"]

agents = ["market_maker(1)", "market_maker(2)", "market_maker(3)",
          "market_taker(1)", "market_taker(2)", "market_taker(3)",
          "fundamentalist(1)", "fundamentalist(2)", "fundamentalist(3)", "fundamentalist(4)",
          "chartist(1)", "chartist(2)", "chartist(3)", "chartist(4)", "noise_trader(1)"]

class NeuralNetwork(nn.Module):
    def __init__(self, l0=18, l1=256, l2=1024, l3=1024, ln=15, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(l0, l1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(l3, ln)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def load_and_preprocess_data(experiment, selected_features):
    # Load data
    x_train = np.loadtxt("../../data/x_train_" + str(experiment) + ".csv")
    y_train = np.loadtxt("../../data/y_train_" + str(experiment) + ".csv", dtype=np.int32)
    x_test = np.loadtxt("../../data/x_test_" + str(experiment) + ".csv")
    y_test = np.loadtxt("../../data/y_test_" + str(experiment) + ".csv", dtype=np.int32)

    # Choose features
    x_train = x_train[:, selected_features]
    x_test = x_test[:, selected_features]

    # Standardize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def predict_nn(x_train, y_train, x_test, y_test, args, return_model=True):
    # Use GPU if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Create data loaders
    train_data = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Create model, loss function and optimizer
    model = NeuralNetwork(
        x_train.shape[1],
        args.l1,
        args.l2,
        args.l3,
        y_train.numpy().max()+1,
        args.dropout
        )
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    # Predict
    model.eval()
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test_pred = model(x_test)
        y_test_pred = y_test_pred.argmax(1).cpu().numpy()
        x_train = x_train.to(device)
        y_train_pred = model(x_train)
        y_train_pred = y_train_pred.argmax(1).cpu().numpy()

    if return_model:
        return y_train_pred, y_test_pred, model
    else:
        return y_train_pred, y_test_pred

def explain_svm(svc, selected_features, n_classes):
    coefs = np.zeros((n_classes, n_classes, len(selected_features)))

    temp = (np.abs(svc.coef_).T / np.abs(svc.coef_).sum(1))
    k = 0
    for i in range(np.max(y_train)+1):
        for j in range(i+1, np.max(y_train)+1):
            coefs[i, j, :] = temp[:, k]
            k += 1

    return coefs

def plot_confusion_matrix(cm, experiment, n_features):
    py.figure(figsize=(5, 4))
    py.imshow(cm, cmap='RdBu', vmin=-1., vmax=1.)
    if experiment == 3:
        cbar = py.colorbar(fraction=0.035)
        cbar.ax.tick_params(labelsize=18)
    py.xticks([])
    py.yticks([])
    py.tick_params(direction='in', top=True, right=True, pad=7)
    py.tight_layout(pad=0.1)
    # fundamentalists
    py.vlines(5.5, ymin=5.5, ymax=9.5, color="gray", ls="--", lw=1)
    py.vlines(9.5, ymin=5.5, ymax=9.5, color="gray", ls="--", lw=1)
    py.hlines(5.5, xmin=5.5, xmax=9.5, color="gray", ls="--", lw=1)
    py.hlines(9.5, xmin=5.5, xmax=9.5, color="gray", ls="--", lw=1)
    # chartsits
    py.vlines(13.5, ymin=9.5, ymax=13.5, color="gray", ls="--", lw=1)
    py.vlines(9.5, ymin=9.5, ymax=13.5, color="gray", ls="--", lw=1)
    py.hlines(13.5, xmin=9.5, xmax=13.5, color="gray", ls="--", lw=1)
    py.hlines(9.5, xmin=9.5, xmax=13.5, color="gray", ls="--", lw=1)
    # market makers
    py.vlines(-0.5, ymin=-0.5, ymax=2.5, color="gray", ls="--", lw=1)
    py.vlines(2.5, ymin=-0.5, ymax=2.5, color="gray", ls="--", lw=1)
    py.hlines(-0.5, xmin=-0.5, xmax=2.5, color="gray", ls="--", lw=1)
    py.hlines(2.5, xmin=-0.5, xmax=2.5, color="gray", ls="--", lw=1)
    # market takers
    py.vlines(2.5, ymin=2.5, ymax=5.5, color="gray", ls="--", lw=1)
    py.vlines(5.5, ymin=2.5, ymax=5.5, color="gray", ls="--", lw=1)
    py.hlines(2.5, xmin=2.5, xmax=5.5, color="gray", ls="--", lw=1)
    py.hlines(5.5, xmin=2.5, xmax=5.5, color="gray", ls="--", lw=1)
    py.savefig("../../plots/confusion/dnn_cm_" +
               str(experiment) + "_" +
               str(n_features) + ".pdf")

def print_classification_report(report, n_classes):
    for i, k in enumerate(report.keys()):
        if i < n_classes:
            print(agents[i], "&",
                f'{report[k]["precision"]:.2f}', "&",
                f'{report[k]["recall"]:.2f}', "&",
                f'{report[k]["f1-score"]:.2f}', "&",
                f'{report[k]["support"]:.0f}', "\\\\")

def plot_explanation(selected_features, coefs, experiment):
    i_max = len(selected_features) // 3

    fig, axs = py.subplots(i_max, 3, figsize=(10, 3 * i_max))
    axs[0, 0].imshow(coefs[:, :, 0], cmap='Blues', vmin=0.0, vmax=1.0)

    for i in range(i_max):
        for j in range(3):
            axs[i, j].set_title(features[i + i_max*j])
            axs[i, j].imshow(coefs[:, :, i + i_max*j], cmap='Blues', vmin=0.0, vmax=1.0)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    py.tight_layout()
    py.savefig("../../plots/explanation/svm_explain_" +
               str(experiment) + "_" +
               str(len(selected_features)) + ".pdf")

if __name__ == "__main__":
    # Choose features
    selected_features = list(range(len(features)))
    selected_features = selected_features[:args.n_features]

    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data(args.experiment, selected_features)
    n_classes = y_train.numpy().max()+1

    # Fit and predict
    y_train_pred, y_test_pred, model = predict_nn(
        x_train,
        y_train,
        x_test,
        y_test,
        args,
        return_model=True
    )
    print("Train accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))

    if args.plot_confusion:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        res_cm = -(cm.T / cm.sum(1)).T
        res_cm = res_cm - 2 * np.diag(np.diag(res_cm))

        plot_confusion_matrix(res_cm, args.experiment, args.n_features)

    if args.print_report:
        # Classification report
        report = classification_report(
            y_test,
            y_test_pred,
            output_dict=True
            )
        print_classification_report(report, n_classes)

    # if args.plot_explanation:
    #     # Explanation
    #     coefs = explain_svm(svc, selected_features, n_classes)
    #     plot_explanation(selected_features, coefs, args.experiment)
