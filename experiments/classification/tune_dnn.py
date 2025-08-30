
import numpy as np
import argparse
import os
import tempfile
import json

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser(description="DNN tuning")

parser.add_argument('-experiment', default=1, dest='experiment', type=int)
parser.add_argument('-n_features', default=18, dest='n_features', type=int)
parser.add_argument('-params', default='dnn_tune_config.json', dest='params', type=str)
parser.add_argument('-max_num_epochs', default=10, dest='max_num_epochs', type=int)
parser.add_argument('-num_samples', default=20, dest='num_samples', type=int)
parser.add_argument('-gpus_per_trial', default=0, dest='gpus_per_trial', type=int)

args = parser.parse_args()

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

def load_data(path, experiment, n_features):
    # Load data
    x_train = np.loadtxt(path + "data/x_train_" + str(experiment) + ".csv")
    y_train = np.loadtxt(path + "data/y_train_" + str(experiment) + ".csv", dtype=np.int32)

    x_validation = np.loadtxt(path + "data/x_validation_" + str(experiment) + ".csv")
    y_validation = np.loadtxt(path + "data/y_validation_" + str(experiment) + ".csv", dtype=np.int32)

    # Choose features
    x_train = x_train[:, :n_features]
    x_validation = x_validation[:, :n_features]

    # Standardize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)

    # Create a TensorDataset
    train_set = TensorDataset(x_train_tensor, y_train_tensor)
    validation_set = TensorDataset(x_validation_tensor, y_validation_tensor)

    return train_set, validation_set

def load_test_data(path, experiment, n_features):
    # Load data
    x_train = np.loadtxt(path + "data/x_train_" + str(experiment) + ".csv")
    x_test = np.loadtxt(path + "data/x_test_" + str(experiment) + ".csv")
    y_test = np.loadtxt(path + "data/y_test_" + str(experiment) + ".csv", dtype=np.int32)

    # Choose features
    x_train = x_train[:, :n_features]
    x_test = x_test[:, :n_features]

    # Standardize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert numpy arrays to PyTorch tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create a TensorDataset
    test_set = TensorDataset(x_test_tensor, y_test_tensor)

    return test_set

def train_agents(config):
    net = NeuralNetwork(
        config["n_features"],
        config["l1"],
        config["l2"],
        config["l3"],
        config["n_classes"],
        config["dropout"]
        )
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    train_subset, validation_subset = load_data(
        config["path"],
        config["experiment"],
        config["n_features"]
        )

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )
    valloader = torch.utils.data.DataLoader(
        validation_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 200 == 199:  # print every 200 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):  # TODO: is the '0' necessary?
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")

def test_best_model(best_result):
    best_trained_model = NeuralNetwork(
        best_result.config["n_features"],
        best_result.config["l1"],
        best_result.config["l2"],
        best_result.config["l3"],
        best_result.config["n_classes"],
        best_result.config["dropout"]
        )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    testset = load_test_data(
        best_result.config["path"],
        best_result.config["experiment"],
        best_result.config["n_features"]
        )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2  # TODO: why batch_size=4?
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Best trial test set accuracy: {}".format(correct / total))

if __name__ == "__main__":
    # Get directory path
    path = os.getcwd()
    path = path[:-26]

    # Load parameters
    with open(path + "configuration/" + args.params) as f:
        parameters = json.load(f)

    config = {
        "experiment": args.experiment,
        "path": path,
        "n_features": args.n_features,
        "n_classes": 14 if args.experiment == 3 else 15,
        "epochs": parameters["epochs"],
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(parameters["l1"][0], parameters["l1"][1])),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(parameters["l2"][0], parameters["l2"][1])),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(parameters["l3"][0], parameters["l3"][1])),
        "dropout": tune.choice(parameters["dropout"]),
        "lr": tune.loguniform(parameters["lr"][0], parameters["lr"][1]),
        "batch_size": tune.choice(parameters["batch_size"])
    }

    scheduler = ASHAScheduler(
        max_t=args.max_num_epochs,
        grace_period=1,
        reduction_factor=2
        )
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_agents),
            resources={"cpu": 2, "gpu": args.gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=args.num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["accuracy"]))

    test_best_model(best_result)
