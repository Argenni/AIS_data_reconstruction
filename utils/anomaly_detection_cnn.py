# ----------- Library of functions used in anomaly detection phase of AIS message reconstruction ----------
import numpy as np
import torch
torch.manual_seed(0)


class ConvNetReg(torch.nn.Module):
    """
    Convolutional neural network for anomaly detection inside clusters:
    decide whether given wavefrom is damaged or not
    """
    _sample_length = 20
    _max_channels = 8
    _kernel_size = 5
    _padding = 0
    _stride = 1
    
    def __init__(self, sample_length=20, max_channels=4, kernel_size=5, padding=0, stride=1):
        super().__init__()
        # Important variables
        self._sample_length = sample_length
        self._max_channels = max_channels
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        layer1_output_size = (self._sample_length+2*self._padding-self._kernel_size)/self._stride+1 #after conv
        layer1_output_size = int((layer1_output_size-2)/2+1) #after maxpool
        layer2_output_size = (layer1_output_size+2*self._padding-self._kernel_size)/self._stride+1 #after conv
        layer2_output_size = int((layer2_output_size-2)/1+1) #after maxpool
        # Neural network layers
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1, 
                out_channels=int(self._max_channels/2), 
                kernel_size=self._kernel_size,
                padding=self._padding,
                stride=self._stride),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
            )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=int(self._max_channels/2), 
                out_channels=self._max_channels, 
                kernel_size=self._kernel_size,
                padding=self._padding,
                stride=self._stride),
            torch.nn.MaxPool1d(kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        ) 
        self.output_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                    in_features=layer2_output_size*self._max_channels, 
                    #out_features=self._sample_length),
                    out_features=1),
            torch.nn.Sigmoid() )

    def forward(self, X):
        X = torch.tensor(np.array(X), dtype=torch.float)
        X = torch.reshape(X, (-1, 1, self._sample_length))
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.output_layer(X)
        return X
        