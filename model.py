import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = functional.relu(out)
        return out



class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class MLP(nn.Module):
    def __init__(self, out_size = 1, in_size=1,hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(in_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer_1 = nn.Linear(hidden_size, 1)
        self.output_layer_2 = nn.Linear(hidden_size, out_size)
        self.bn = nn.BatchNorm1d(in_size)


    def forward(self, x):
        x = self.bn(x)
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x1 = self.output_layer_1(x)
        x1 = torch.tanh(x1)
        # x2 = self.output_layer_2(x)
        # x2 = torch.nn.functional.softmax(x2)
        return x1