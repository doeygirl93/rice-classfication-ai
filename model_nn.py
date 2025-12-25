import torch.nn as nn

def define_nn_arch(HIDDEN_NURONS, input_size):
    class Net(nn.Module):

        def __init__(self):

            super(Net, self).__init__()

            self.input_layer = nn.Linear(input_size, HIDDEN_NURONS)
            self.relu = nn.ReLU()

            self.output_layer = nn.Linear(HIDDEN_NURONS, 1)

            self.sigmoid = nn.Sigmoid()


        def forward(self, x):
            x = self.input_layer(x)
            x = self.relu(x)
            x = self.output_layer(x)
            x = self.sigmoid(x)
            return x
    return Net()
