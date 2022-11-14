import torch
import torch.nn as nn
import torch.functional as F

class LinearRegres(nn.Module):
    """ Simple linear regressor """
    def __init__(self, ROIs):
        super().__init__()
        features = ROIs*(ROIs-1)//2 # Lower triangular connections
        self.layer = nn.Linear(in_features=features, out_features=features, dtype=torch.float64)
    
    def forward(self, x):
        return self.layer(x)

class NonLinearRegres(nn.Module):
    """ Simple linear regressor """
    def __init__(self, ROIs, ampli=1/2):
        super().__init__()
        features = ROIs*(ROIs-1)//2 # Lower triangular connections
        self.multi_layer = nn.Sequential(
            nn.Linear(in_features=features, out_features=int(features*ampli), dtype=torch.float64),
            nn.Sigmoid(),
            nn.Linear(in_features=int(features*ampli), out_features=int(features*ampli), dtype=torch.float64),
            nn.Sigmoid(),
            nn.Linear(in_features=int(features*ampli), out_features=features, dtype=torch.float64)
        )
    
    def forward(self, x):
        return self.multi_layer(x)