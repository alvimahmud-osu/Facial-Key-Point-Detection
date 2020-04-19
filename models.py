## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = torch.nn.Conv2d(1,32,5) # (32,220,220) output tensor # (W-F)/S + 1 = (224-5)/1 + 1 = 220
        
        # first Max-pooling layer
        self.pool1 = torch.nn.MaxPool2d(4,4) # (32,55,55) output tensor
        
        # second convolutional layer
        self.conv2 = torch.nn.Conv2d(32,64,5) # (64,51,51) output tensor # (W-F)/S + 1 = (55-5)/1 + 1 = 51
        
        # second Max-pooling layer
        self.pool2 = torch.nn.MaxPool2d(4,4) # (64,12,12) output tensor
        
       
        
        # Fully connected layer
        self.fc1 = torch.nn.Linear(64*12*12, 1000)   
        self.fc2 = torch.nn.Linear(1000, 500)       
        self.fc3 = torch.nn.Linear(500, 136)        
        
        self.drop1 = nn.Dropout2d(p=0.4)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        
        
        x = self.pool2(F.relu(self.conv2(x)))
       
        
        
        # Flatten before passing to the fully-connected layers.
        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        
        x = F.relu(x)
        
        
        x = self.fc2(x)
        x = F.relu(x)
       
        x = self.fc3(x)
        
        x = self.drop1(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x