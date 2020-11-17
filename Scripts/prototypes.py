# Indoor Semantic Segmentation
#
# Date: 15/11/2020
# Authors: Xavier Riera, Albert Mestre, Jose Javier Gomez
# Institute: Universitat Politecnica De Catalunya
#

import torch
import torch.nn as nn


#UNET Convolutional block (2 conv, 2 batchnorm and 2 ReLU )
class ConvBlock(nn.Module):
    
    def __init__(self, num_imp_channels=3, num_out_fmaps=1, kernel_conv=3 , dropout1 = 0, dropout2 = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(num_imp_channels, num_out_fmaps, kernel_conv, padding=1)
        self.norm1 = nn.BatchNorm2d(num_out_fmaps)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout1)
        self.conv2 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=1)
        self.norm2 = nn.BatchNorm2d(num_out_fmaps)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout2)

    def forward(self, x): 
        return self.drop2(self.relu2(self.norm2(self.conv2(self.drop1(self.relu1(self.norm1(self.conv1(x))))))))




#PD ADAM LR5 WD4 BATCH04 FEATURESBOTTLENECK256 Training.ipynb  - 3 Skip Connections 256 features
#
#Unet Model with (RGB Image input and 13 predicted classes of output)
class ParallelDilatedConvolutionModule_SkipConnections_3_Features_256(nn.Module):
    def __init__(self, num_imp_channels, num_out_fmaps, kernel_conv=3, dropout = 0, addition = 1):
        super().__init__()
        
        self.addition = addition

        self.conv1 = nn.Conv2d(num_imp_channels, num_out_fmaps, kernel_conv, padding=1, dilation=1 )
        self.norm1 = nn.BatchNorm2d(num_out_fmaps)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=2, dilation=2 )
        self.norm2 = nn.BatchNorm2d(num_out_fmaps)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=4, dilation=4 )
        self.norm3 = nn.BatchNorm2d(num_out_fmaps)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=8, dilation=8 )        
        self.norm4 = nn.BatchNorm2d(num_out_fmaps)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=16, dilation=16 )        
        self.norm5 = nn.BatchNorm2d(num_out_fmaps)
        self.relu5 = nn.ReLU()
      

    def forward(self, x): 

        dilated1 = self.dropout1(self.relu1(self.norm1(self.conv1(x))) ) 
        dilated2 = self.dropout2(self.relu2(self.norm2(self.conv2(dilated1))) )
        dilated3 = self.dropout3(self.relu3(self.norm3(self.conv3(dilated2))) )
        dilated4 = self.dropout4(self.relu4(self.norm4(self.conv4(dilated3))) )
        dilated5 = self.relu5(self.norm5(self.conv5(dilated4)))
        
        if self.addition == 1:
        	return (dilated1 + dilated2 + dilated3 + dilated4 + dilated5)
        else:
          return dilated5

#Unet Model with (RGB Image input and 13 predicted classes of output)
class DeepDilatedUnet_SkipConnections_3_Features_256(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32, n_class=13, dropout =0, addition = 1):
        super().__init__()

        #Encoder
        self.encoder1 = ConvBlock(3, 32, dropout1 = dropout, dropout2 = dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(dropout)
        self.encoder2 = ConvBlock(32, 64, dropout1 = dropout, dropout2 = dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(dropout)
        self.encoder3 = ConvBlock(64, 128, dropout1 = dropout, dropout2 = dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(dropout)

        #NeckBottle
        self.neck1 = ParallelDilatedConvolutionModule_SkipConnections_3_Features_256(128, 256, 3, dropout, addition)

        #Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(dropout)
        self.decoder3 = ConvBlock(128*2, 128, dropout1 = dropout, dropout2 = dropout)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
       	self.drop5 = nn.Dropout(dropout) 
       	self.decoder2 = ConvBlock(64*2, 64, dropout1 = dropout, dropout2 = dropout)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
       	self.drop6 = nn.Dropout(dropout) 
        self.decoder1 = ConvBlock(32*2, 32)
        self.conv = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        down1 = self.encoder1(x)
        down1pool = self.drop1(self.pool1(down1))
        down2 = self.encoder2(down1pool)
        down2pool = self.drop2(self.pool2(down2))
        down3 = self.encoder3(down2pool)
        down3pool = self.drop3(self.pool3(down3))
        neck1 = self.neck1(down3pool)

        upc3 = self.drop4(self.upconv3(neck1))
        concat3 = torch.cat((down3, upc3),dim=1)
        up3 = self.decoder3(concat3)

        upc2 = self.drop5(self.upconv2(up3))
        concat2 = torch.cat((down2, upc2),dim=1)
        up2 = self.decoder2(concat2)

        upc1 = self.drop6(self.upconv1(up2))
        concat1 = torch.cat((down1, upc1),dim=1)
        up1 = self.decoder1(concat1)

        x = self.conv(up1)

        return x
		
		
		
#PD_ADAM_LR5_WD_DROP_0_BATCH04_SKIPCONNECT4_BOTTLENECK512 Training.ipynb - 4 Skip Connections 512 features
#
#Unet Model with (RGB Image input and 13 predicted classes of output)
class ParallelDilatedConvolutionModule_SkipConnections_4_Features_512(nn.Module):
    def __init__(self, num_imp_channels, num_out_fmaps, kernel_conv=3, dropout = 0, addition = 1):
        super().__init__()
        
        self.addition = addition

        self.conv1 = nn.Conv2d(num_imp_channels, num_out_fmaps, kernel_conv, padding=1, dilation=1 )
        self.norm1 = nn.BatchNorm2d(num_out_fmaps)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=2, dilation=2 )
        self.norm2 = nn.BatchNorm2d(num_out_fmaps)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=4, dilation=4 )
        self.norm3 = nn.BatchNorm2d(num_out_fmaps)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=8, dilation=8 )        
        self.norm4 = nn.BatchNorm2d(num_out_fmaps)
        self.relu4 = nn.ReLU()
      

    def forward(self, x): 

        dilated1 = self.dropout1(self.relu1(self.norm1(self.conv1(x))) ) 
        dilated2 = self.dropout2(self.relu2(self.norm2(self.conv2(dilated1))) )
        dilated3 = self.dropout3(self.relu3(self.norm3(self.conv3(dilated2))) )
        dilated4 = self.relu4(self.norm4(self.conv4(dilated3)))
        
        if self.addition == 1:
        	return (dilated1 + dilated2 + dilated3 + dilated4 )
        else:
          return dilated4

#Unet Model with (RGB Image input and 13 predicted classes of output)
class DeepDilatedUnet_SkipConnections_4_Features_512(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32, n_class=13, dropout =0, addition = 1):
        super().__init__()

        #Encoder
        self.encoder1 = ConvBlock(3, 32, dropout1 = dropout, dropout2 = dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(dropout)
        self.encoder2 = ConvBlock(32, 64, dropout1 = dropout, dropout2 = dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(dropout)
        self.encoder3 = ConvBlock(64, 128, dropout1 = dropout, dropout2 = dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(dropout)

        self.encoder4 = ConvBlock(128, 256, dropout1 = dropout, dropout2 = dropout)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(dropout)



        #NeckBottle
        self.neck1 = ParallelDilatedConvolutionModule_SkipConnections_4_Features_512(256, 512, 3, dropout, addition)

        #Decoder

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.drop5 = nn.Dropout(dropout)
        self.decoder4 = ConvBlock(256*2, 256, dropout1 = dropout, dropout2 = dropout)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.drop6 = nn.Dropout(dropout)
        self.decoder3 = ConvBlock(128*2, 128, dropout1 = dropout, dropout2 = dropout)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
       	self.drop7 = nn.Dropout(dropout) 
       	self.decoder2 = ConvBlock(64*2, 64, dropout1 = dropout, dropout2 = dropout)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
       	self.drop8 = nn.Dropout(dropout) 
        self.decoder1 = ConvBlock(32*2, 32)
        self.conv = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        down1 = self.encoder1(x)
        down1pool = self.drop1(self.pool1(down1))
        down2 = self.encoder2(down1pool)
        down2pool = self.drop2(self.pool2(down2))
        down3 = self.encoder3(down2pool)
        down3pool = self.drop3(self.pool3(down3))
        down4 = self.encoder4(down3pool)
        down4pool = self.drop4(self.pool4(down4))


        neck1 = self.neck1(down4pool)

        upc4 = self.drop5(self.upconv4(neck1))
        concat4 = torch.cat((down4, upc4),dim=1)
        up4 = self.decoder4(concat4)


        upc3 = self.drop6(self.upconv3(up4))
        concat3 = torch.cat((down3, upc3),dim=1)
        up3 = self.decoder3(concat3)

        upc2 = self.drop7(self.upconv2(up3))
        concat2 = torch.cat((down2, upc2),dim=1)
        up2 = self.decoder2(concat2)

        upc1 = self.drop8(self.upconv1(up2))
        concat1 = torch.cat((down1, upc1),dim=1)
        up1 = self.decoder1(concat1)

        x = self.conv(up1)

        return x


#PD_ADAM_LR5_WD_DROP_2_BATCH16_DILATIONS6 Training.ipynb - 3 Skip Connections 512 features 1,2,3,4,5 dilations
#

#Unet Model with (RGB Image input and 13 predicted classes of output)
class ParallelDilatedConvolutionModule_SkipConnections_3_Features_512_Dilations_123456(nn.Module):
    def __init__(self, num_imp_channels, num_out_fmaps, kernel_conv=3, dropout = 0, addition = 1):
        super().__init__()
        
        self.addition = addition

        self.conv0 = nn.Conv2d(num_imp_channels, num_out_fmaps, kernel_conv, padding=1)
        self.norm0 = nn.BatchNorm2d(num_out_fmaps)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=1, dilation=1 )
        self.norm1 = nn.BatchNorm2d(num_out_fmaps)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=2, dilation=2 )
        self.norm2 = nn.BatchNorm2d(num_out_fmaps)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=3, dilation=3 )
        self.norm3 = nn.BatchNorm2d(num_out_fmaps)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=4, dilation=4 )        
        self.norm4 = nn.BatchNorm2d(num_out_fmaps)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
		
        self.conv5 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=5, dilation=5 )        
        self.norm5 = nn.BatchNorm2d(num_out_fmaps)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout)

        self.conv6 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=6, dilation=6 )        
        self.norm6 = nn.BatchNorm2d(num_out_fmaps)
        self.relu6 = nn.ReLU()

    def forward(self, x): 

        x = self.relu0(self.norm0(self.conv0(x)))
        dilated1 = self.dropout1(self.relu1(self.norm1(self.conv1(x))) ) 
        dilated2 = self.dropout2(self.relu2(self.norm2(self.conv2(dilated1))) )
        dilated3 = self.dropout3(self.relu3(self.norm3(self.conv3(dilated2))) )
        dilated4 = self.dropout4(self.relu4(self.norm4(self.conv4(dilated3))) )
        dilated5 = self.dropout5(self.relu5(self.norm5(self.conv5(dilated4))) )
        dilated6 = self.relu6(self.norm6(self.conv6(dilated5)))
		
        if self.addition == 1:
        	return (dilated1 + dilated2 + dilated3 + dilated4 + dilated5 + dilated6)
        else:
          return dilated6

#Unet Model with (RGB Image input and 13 predicted classes of output)
class DeepDilatedUnet_SkipConnections_3_Features_512_Dilations_123456(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32, n_class=13, dropout =0, addition = 1):
        super().__init__()

        #Encoder
        self.encoder1 = ConvBlock(3, 64, dropout1 = dropout, dropout2 = dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(dropout)
        self.encoder2 = ConvBlock(64, 128, dropout1 = dropout, dropout2 = dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(dropout)
        self.encoder3 = ConvBlock(128, 256, dropout1 = dropout, dropout2 = dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(dropout)

        #NeckBottle
        self.neck1 = ParallelDilatedConvolutionModule_SkipConnections_3_Features_512_Dilations_123456(256, 512, 3, dropout, addition)

        #Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(dropout)
        self.decoder3 = ConvBlock(256*2, 256, dropout1 = dropout, dropout2 = dropout)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
       	self.drop5 = nn.Dropout(dropout) 
       	self.decoder2 = ConvBlock(128*2, 128, dropout1 = dropout, dropout2 = dropout)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
       	self.drop6 = nn.Dropout(dropout) 
        self.decoder1 = ConvBlock(64*2, 64)
        self.drop9 = nn.Dropout(0.2)
        self.conv = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        down1 = self.encoder1(x)
        down1pool = self.drop1(self.pool1(down1))
        down2 = self.encoder2(down1pool)
        down2pool = self.drop2(self.pool2(down2))
        down3 = self.encoder3(down2pool)
        down3pool = self.drop3(self.pool3(down3))
        neck1 = self.neck1(down3pool)

        upc3 = self.drop4(self.upconv3(neck1))
        concat3 = torch.cat((down3, upc3),dim=1)
        up3 = self.decoder3(concat3)

        upc2 = self.drop5(self.upconv2(up3))
        concat2 = torch.cat((down2, upc2),dim=1)
        up2 = self.decoder2(concat2)

        upc1 = self.drop6(self.upconv1(up2))
        concat1 = torch.cat((down1, upc1),dim=1)
        up1 = self.drop9(self.decoder1(concat1))

        x = self.conv(up1)

        return x


