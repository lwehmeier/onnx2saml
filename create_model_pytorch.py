import torch.nn as nn
import torch.nn.init as init
import torch.onnx


class ForwardDenseReluModel(nn.Module):
    def __init__(self, inplace=False):
        super(ForwardDenseReluModel, self).__init__()

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
class RnnDenseReluModel(nn.Module):
    def __init__(self, inplace=False):
        super(RnnDenseReluModel, self).__init__()
        self.lin1 = nn.RNN(11, 10, nonlinearity='relu', batch_first=True)
        self.lin2 = nn.RNN(10, 7, nonlinearity='relu', batch_first=True)
        self.lin3 = nn.RNN(7, 5, nonlinearity='relu', batch_first=True)

        self._initialize_weights()

    def forward(self, x, h):

        x,_ = self.lin1(x,h[0])
        x,_ = self.lin2(x,h[1])
        x,_ = self.lin3(x,h[2])
        return x

    def _initialize_weights(self):
            for m in self.modules():
                if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            torch.nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            torch.nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            param.data.fill_(0)

torch_model = ForwardDenseReluModel()
torch_model.train(False)
x = torch.randn(1, 11, requires_grad=True)
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "data/demo_model.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file


torch_model = RnnDenseReluModel()
torch_model.train(False)
x = torch.randn(1,25, 11, requires_grad=True)
h = [torch.zeros((1,1,10)),torch.zeros((1,1,7)),torch.zeros((1,1,5))]
torch_model.forward(x,h)
torch_out = torch.onnx._export(torch_model,             # model being run
                               (x,h),                       # model input (or a tuple for multiple inputs)
                               "data/demo_model_rnn.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file