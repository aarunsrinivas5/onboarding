from torch import nn
import torch.nn.functional as F

class DNN(nn.Module):
    
    def __init__(self, input_dim, fc1_dim, fc2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.output = nn.Linear(fc2_dim, output_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.output(x)
        return x
    