# ----------- Library of neural nets used in AIS message reconstruction ----------
import numpy as np
import torch
torch.manual_seed(0)


class NeuralNet(torch.nn.Module):
    """
    Neural network for anomaly detection inside clusters:
    decide whether given message is damaged or not and which field is damaged
    """
    _in_features = 14
    _max_features = 42
    _classes = 6
    
    def __init__(self, goal, in_features=14, max_features=42, classes = 6):
        super().__init__()
        # Important variables
        self._in_features = in_features
        self._max_features = max_features
        self._classes = classes
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
        if goal=="binary":
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=self._max_features, out_features=1),
                torch.nn.Sigmoid() )
        if goal=="multi_label":
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=self._max_features, out_features=self._classes),
                torch.nn.Sigmoid() )

    def forward(self, X):
        X = torch.tensor(np.array(X), dtype=torch.float)
        X = torch.reshape(X, (-1, self._in_features))
        X = self.layer1(X)
        X = self.layer2(X)
        out = self.output_layer(X)
        return out
        