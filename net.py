import torch.nn as nn
import torch.nn.functional as F
import torch

class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6,15,6)
        self.fc1 = nn.Linear(15,32)#16*4*4
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 10)
        self.AA2D = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #x = x.view(-1, 16*4*4)
        x = self.AA2D(x)
        x = torch.squeeze(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net =MNet()
    print(net)
