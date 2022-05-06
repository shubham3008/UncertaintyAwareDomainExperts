import torch.nn as nn
import torch.nn.functional as F
import torch 

from dropouts import DROPOUT_FACTORY


class LinearNetwork(nn.Module):
  def __init__(self, input_size, output_size, layers, activation, args):
    super(LinearNetwork, self).__init__()
    self.args = args
    self.input_size = 1

    for i in input_size:
        self.input_size*=int(i) 
    self.output_size = int(output_size)

    if activation == "relu":
        self.activation = nn.ReLU
    elif activation == "tanh":
        self.activation = nn.Tanh
    elif activation == "linear":
        self.activation = nn.Identity
    else:
        raise NotImplementedError("Other activations have not been implemented!")

    self.layers = nn.ModuleList()
    for i in range(len(layers)):
        if i == 0:
            self.layers.append(nn.Linear(self.input_size, int(layers[0]),  bias=False))
        else:
            self.layers.append(nn.Linear(int(layers[i-1]), int(layers[i]),  bias=False))
        self.layers.append(self.activation())
        self.layers.append(DROPOUT_FACTORY[self.args.dropout_type](self.args.p))


    self.layers.append(nn.Linear(int(layers[len(layers)-1]), self.output_size,  bias=False))
    
    
  def forward(self, input):
    x = input.view(-1,self.input_size)
    for i, layer in enumerate(self.layers):
        x = layer(x)

    if self.args.task == "binary_classification":
        x = torch.sigmoid(x)
    elif self.args.task == "classification":
        x = F.softmax(x, dim=-1)
    return x, torch.tensor([0.0]).view(1).to(input.device)
  

  def log(self, *args, **kwargs):
    pass
  

class ConvNetwork(nn.Module):
    def __init__(self, input_size, output_size, layers, activation, args):
        super(ConvNetwork, self).__init__()
        self.args = args 

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "linear":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError("Other activations have not been implemented!")

        self.dropout = DROPOUT_FACTORY[self.args.dropout_type](self.args.p)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, padding=2,  bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2,  bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, padding=2,  bias=False)
        self.linear1 = nn.Linear(in_features=3200, out_features=100,  bias=False)
        self.linear2 = nn.Linear(in_features=100, out_features=output_size,  bias=False)

        # self.network = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

        #     nn.Flatten(), 
        #     nn.Linear(256*4*4, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10))


    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # x = self.network(x)
        x = F.softmax(x, dim=-1)
        return x

    def log(self, *args, **kwargs):
        pass