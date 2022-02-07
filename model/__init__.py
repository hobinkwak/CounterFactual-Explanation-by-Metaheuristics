import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 10)
        self.fc5 = nn.Linear(10, n_outputs)

    def forward(self, x):

        x = x.float()
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = self.fc5(h4)
        return h5

if __name__ == '__main__':
    pass