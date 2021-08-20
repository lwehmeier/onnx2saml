# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.lin1 = nn.Linear(11, 10)
        self.lin2 = nn.Linear(10, 7)
        self.lin3 = nn.Linear(7, 5)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.lin1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.lin2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.lin3.weight, init.calculate_gain('relu'))

# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet()

import torch.onnx
# set the train mode to false since we will only run the forward pass.
torch_model.train(False)
# Input to the model
x = torch.randn(1, 11, requires_grad=True)

# Export the model
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "data/demo_model.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file