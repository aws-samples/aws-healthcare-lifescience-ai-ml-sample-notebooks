import torch.nn as nn

class SurvivalModel(nn.Module):
    
    def __init__(self, n_input_dim=21, n_hidden1 = 300, n_hidden2 = 100, n_output =  1):
        
        super(SurvivalModel, self).__init__()
        self.n_input_dim = n_input_dim
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output)
        
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x

    def serialize_params(self):
        return {
            "model": {
                "n_input_dim": self.n_input_dim
            }
        }