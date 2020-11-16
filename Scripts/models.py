# Indoor Semantic Segmentation
#
# Date: 15/11/2020
# Authors: Xavier Riera, Albert Mestre, Jose Javier Gomez
# Institute: Universitat Politecnica De Catalunya
#

import torch
import torch.nn as nn

#Convolutional block (2 conv, 2 batchnorm and 2 ReLU to be clean the code in the UNET)
class ConvBlock(nn.Module):
    
    def __init__(self, num_imp_channels=3, num_out_fmaps=1, kernel_conv=3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_imp_channels, num_out_fmaps, kernel_conv, padding=1)
        self.norm1 = nn.BatchNorm2d(num_out_fmaps)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=1)
        self.norm2 = nn.BatchNorm2d(num_out_fmaps)
        self.relu2 = nn.ReLU()
    def forward(self, x): 
        return self.relu2(self.norm2(self.conv2(self.relu1(self.norm1(self.conv1(x))))))

#Unet Model with (RGB Image input and 13 predicted classes of output)
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32, n_class=13, dropout = 0):
        super().__init__()
        
        #Encoder
        self.encoder1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #NeckBottle
        self.neck1 = ConvBlock(256,512)

        #Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(256*2, 256) #2 channels of 256
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(128*2, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(64*2, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(32*2, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.conv = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        neck1 = self.neck1(self.pool4(enc4))
        concat4 = torch.cat((self.upconv4(neck1),enc4),dim=1)
        dec4 = self.decoder4(concat4)
        concat3 = torch.cat((self.upconv3(dec4),enc3),dim=1)
        dec3 = self.decoder3(concat3)
        concat2 = torch.cat((self.upconv2(dec3),enc2),dim=1)
        dec2 = self.decoder2(concat2)
        concat1 = torch.cat((self.upconv1(dec2),enc1),dim=1)
        dec1 = self.dropout1(self.decoder1(concat1))
        return self.conv(dec1)
