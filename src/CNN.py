# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    '''
    Custom CNN architecture
    Note that this has not been optimised for the task
    '''
    def __init__(self):
        ''
        ''
        super().__init__()
        
        # conv layers than do not change size of image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding="same")

        # pooling layer than reduces both height and width of image by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 37 * 37, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        '''
        Pass batch of images through CNN layers until we have predictions for each image
        The predictions are not passed throguh a Sigmoid layer here and so are not guaranteed to be between 0 and 1
        Instead, we use use BCEWithLogitsLoss later for stability
        
        :param x: tensor of size [batch_size, 3, 150, 150]
        
        :returns: tensor of size [batch_size, 1], matching size of labels tensor
        '''
        # x is now 3 channel (RGB) 150 x 150 pixel image
        x = self.pool(F.relu(self.conv1(x)))
        # x is now 6 channel (conv) 75 x 75 pixel image (pool)
        x = self.pool(F.relu(self.conv2(x)))
        # x is now 16 channel (conv) 37 x 37 pixel image (pool)
        
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.float()  # do not use sigmoid here as we use bcewithlogitsloss later


def main():
    '''
    Examine CNN architecture
    '''
    net = CNN()
    print(net)


if __name__ == "__main__":
    
    main()
    