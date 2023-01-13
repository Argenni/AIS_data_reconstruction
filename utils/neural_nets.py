# ----------- Library of neural nets used in AIS message reconstruction ----------
import numpy as np
import torch
torch.manual_seed(0)
import pickle
import matplotlib.pyplot as plt


class NeuralNet(torch.nn.Module):
    """
    Neural network for anomaly detection inside clusters:
    decide whether given message is damaged or not and which field is damaged
    """
    _in_features = 10
    _max_features = 42
    
    def __init__(self, in_features=10, max_features=42):
        super().__init__()
        # Important variables
        self._in_features = in_features
        self._max_features = max_features
        # Neural network layers
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._in_features, out_features=int(self._max_features/2)),
            torch.nn.BatchNorm1d(int(self._max_features/2), track_running_stats=False, affine=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3) )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=int(max_features/2), out_features=max_features),
            torch.nn.BatchNorm1d(self._max_features, track_running_stats=False, affine=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3) )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=max_features, out_features=int(max_features/2)),
            torch.nn.BatchNorm1d(self._max_features, track_running_stats=False, affine=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3) )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=int(self._max_features/2), out_features=4),
            torch.nn.Sigmoid() )

    def forward(self, X):
        X = torch.tensor(np.array(X), dtype=torch.float)
        X = torch.reshape(X, (-1, self._in_features))
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        out = self.output_layer(X)
        return out
        
def train_nn():
    nn = NeuralNet()
    nn = nn.float()
    variables = pickle.load(open('utils/anomaly_detection_files/nn_inputs.h5', 'rb'))
    x_train = variables[0] 
    y_train = variables[1]
    x_val = variables[2]
    y_val = variables[3]
    # Set criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=nn.parameters(), lr=0.001)
    # Train NN
    loss_train = []
    loss_val = []
    for epoch in range(1000):
        # Eval
        nn.eval()
        with torch.no_grad():
            pred = nn(x_val)
            loss = criterion(pred, torch.tensor(y_val, dtype=torch.float))
            loss_val.append(loss.detach().numpy())
        # Train
        nn.train()
        optimizer.zero_grad()
        pred = nn(x_train)
        loss = criterion(pred, torch.tensor(y_train, dtype=torch.float))
        loss.backward()
        optimizer.step()
        print("Epoch " + str(epoch) + ": loss " + str(loss.detach().numpy()))
        loss_train.append(loss.detach().numpy())
    print("  Complete.")
    fig, ax = plt.subplots()
    ax.plot(loss_train, color='k')
    ax.plot(loss_val, color='r')
    ax.set_title("Losses in each epoch - NN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Training loss", "Validation loss"])
    fig.show()