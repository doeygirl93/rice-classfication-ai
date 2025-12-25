### just define the nn and return the nn class'
import torch.nn as nn





def define_nn_arch(HIDDEN_NURONS, X):
    class Net(nn.Module):

        def __init__(self):

            super(Net, self).__init__()

            self.input_layer = nn.Linear(X.shape[1], HIDDEN_NURONS)
            self.relu = nn.ReLU()
            self.linear = nn.Linear(HIDDEN_NURONS, 1) #1 is num of classes (bc binary its 1)
            self.sigmoid = nn.Sigmoid()
            self.linear2 = nn.Linear(HIDDEN_NURONS, 1)

        def forward(self, x):
            x = self.input_layer(x)
            x = self.relu(x)
            x = self.linear(x)
            x = self.sigmoid(x)
            x = self.linear2(x)
            return x
    return Net()
