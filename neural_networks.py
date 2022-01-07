import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import classification_report
import numpy as np
import preprocess
from preprocess import preprocessing

torch.manual_seed(preprocess.seed)
np.random.seed(preprocess.seed)


class Net(nn.Module):
    def __init__(self, INPUT_SHAPE, OUTPUT_SIZE):
        super().__init__()
        self.INPUT_SHAPE = INPUT_SHAPE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.fc1 = nn.Linear(self.INPUT_SHAPE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, self.OUTPUT_SIZE)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        x = F.selu(self.fc5(x))
        return F.log_softmax(x, dim=1)  # F.sigmoid(x


class Net1(nn.Module):
    def __init__(self, INPUT_SHAPE, OUTPUT_SIZE):
        super().__init__()
        self.INPUT_SHAPE = INPUT_SHAPE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.fc1 = nn.Linear(self.INPUT_SHAPE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, self.OUTPUT_SIZE)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc4(x))
        x = F.selu(self.fc5(x))
        return F.log_softmax(x, dim=1)  # F.sigmoid(x


class Net2(nn.Module):
    def __init__(self, INPUT_SHAPE, OUTPUT_SIZE):
        super().__init__()
        self.INPUT_SHAPE = INPUT_SHAPE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.fc1 = nn.Linear(self.INPUT_SHAPE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, self.OUTPUT_SIZE)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc5(x))
        return F.log_softmax(x, dim=1)  # F.sigmoid(x


class Net3(nn.Module):
    def __init__(self, INPUT_SHAPE, OUTPUT_SIZE):
        super().__init__()
        self.INPUT_SHAPE = INPUT_SHAPE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.fc1 = nn.Linear(self.INPUT_SHAPE, 64)
        self.fc5 = nn.Linear(64, self.OUTPUT_SIZE)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc5(x))
        return F.log_softmax(x, dim=1)  # F.sigmoid(x


def train_model(
    net,
    trainset,
    learn_rate,
    loss_function=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    epocs=50,
):
    optimizer = optimizer(net.parameters(), lr=learn_rate)
    for epoch in range(epocs):  # 3 full passes over the data
        for data in trainset:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(
                X.view(-1, X.shape[1])
            )  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = loss_function(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print(
            f"Loss on Epoc: {epoch} is: {loss}"
        )  # print loss. We hope loss (a measure of wrong-ness) declines!


def model_performance(INPUT_SHAPE, net, X_train, y_train, set_type="Training Set"):
    correct = 0
    total = 0
    prediction = torch.tensor([])
    prediction_argmax = []
    with torch.no_grad():
        output = net(X_train.view(-1, INPUT_SHAPE))
        for idx, i in enumerate(output):
            prediction = torch.cat((prediction, i))
            prediction_argmax = np.append(prediction_argmax, torch.argmax(i))
            if torch.argmax(i) == y_train[idx]:
                correct += 1
            total += 1
    print(f"Accuracy on {set_type}: ", round(correct / total, 3))
    print(classification_report(prediction_argmax, y_train))
    return classification_report(prediction_argmax, y_train)

