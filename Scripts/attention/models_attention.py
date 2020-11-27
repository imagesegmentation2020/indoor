import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_grid_gating_signal import UnetGridGatingSignal2
from grid_attention_layer import GridAttentionBlock2D, GridAttentionBlock2D_TORR
from networks_other import init_weights


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
    def __init__(self, feature_scale = 4, in_channels=3, out_channels=1, features=32, n_class=13, dropout = 0, nonlocal_mode='concatenation', attention_dsample=(1,1), is_batchnorm=True, is_2D_TORR = True, init_type = None ):
        super().__init__()

        self.is_batchnorm = is_batchnorm
        self.init_type = init_type
        self.nonlocal_mode = nonlocal_mode

        if is_2D_TORR and self.nonlocal_mode  == 'concatenation':
          self.nonlocal_mode = 'concatenation_mean_flow'

        filters = [32, 64, 128, 256, 512]
        filters = [int(x / feature_scale) for x in filters]

        #Encoder
        self.encoder1 = ConvBlock(3, filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #NeckBottle
        self.neck1 = ConvBlock(filters[3],filters[4])
        self.gating = UnetGridGatingSignal2(filters[4], filters[3], kernel_size=(1, 1), is_batchnorm=self.is_batchnorm, init_type = self.init_type)


        # attention blocks
        if is_2D_TORR :

          self.attentionblock2 = GridAttentionBlock2D_TORR(in_channels=filters[1], gating_channels=filters[3],
                                                      inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode, 
                                                      use_W=False, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu',
                                                      init_type = self.init_type)
          self.attentionblock3 = GridAttentionBlock2D_TORR(in_channels=filters[2], gating_channels=filters[3],
                                                      inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode, 
                                                      use_W=False, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu',
                                                      init_type = self.init_type)
          self.attentionblock4 = GridAttentionBlock2D_TORR(in_channels=filters[3], gating_channels=filters[3],
                                                      inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode, 
                                                      use_W=False, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu',
                                                      init_type = self.init_type)
        else :

          self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[3],
                                                      inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode,
                                                      init_type = self.init_type)
          self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                      inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode,
                                                      init_type = self.init_type)
          self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[3],
                                                      inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode,
                                                      init_type = self.init_type)

        #Decoder
        self.upconv4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(filters[3]*2, filters[3]) #2 channels of filters[3]
        self.upconv3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(filters[2]*2, filters[2])
        self.upconv2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(filters[1]*2, filters[1])
        self.upconv1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(filters[0]*2, filters[0])
        self.dropout1 = nn.Dropout(dropout)
        self.conv = nn.Conv2d(filters[0], n_class, 1)

        if self.init_type : 
          # initialise weights
          for m in self.modules():
              if isinstance(m, nn.Conv3d):
                  init_weights(m, init_type=self.init_type)
              elif isinstance(m, nn.BatchNorm3d):
                  init_weights(m, init_type=self.init_type)



    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        neck1 = self.neck1(self.pool4(enc4))
        
        # Gating Signal Generation
        gating = self.gating(neck1)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(enc4, gating)
        g_conv3, att3 = self.attentionblock3(enc3, gating)
        g_conv2, att2 = self.attentionblock2(enc2, gating)

        concat4 = torch.cat( (g_conv4, self.upconv4(neck1)) ,dim=1)
        dec4 = self.decoder4(concat4)
        concat3 = torch.cat((g_conv3, self.upconv3(dec4)),dim=1)
        dec3 = self.decoder3(concat3)
        concat2 = torch.cat((g_conv2, self.upconv2(dec3)),dim=1)
        dec2 = self.decoder2(concat2)
        concat1 = torch.cat((enc1, self.upconv1(dec2)),dim=1)
        dec1 = self.dropout1(self.decoder1(concat1))
        return self.conv(dec1)


class ParallelDilatedConvolutionModule(nn.Module):
    def __init__(self, num_imp_channels, num_out_fmaps, kernel_conv=3):
        super().__init__()

        self.conv0 = nn.Conv2d(num_imp_channels, num_out_fmaps, kernel_conv, padding=1)
        self.norm0 = nn.BatchNorm2d(num_out_fmaps)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=1, dilation=1 )
        self.norm1 = nn.BatchNorm2d(num_out_fmaps)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=2, dilation=2 )
        self.norm2 = nn.BatchNorm2d(num_out_fmaps)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=3, dilation=3 )
        self.norm3 = nn.BatchNorm2d(num_out_fmaps)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=4, dilation=4 )        
        self.norm4 = nn.BatchNorm2d(num_out_fmaps)
        self.relu4 = nn.ReLU()
		
        self.conv5 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=5, dilation=5 )        
        self.norm5 = nn.BatchNorm2d(num_out_fmaps)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(num_out_fmaps, num_out_fmaps, kernel_conv, padding=6, dilation=6 )        
        self.norm6 = nn.BatchNorm2d(num_out_fmaps)
        self.relu6 = nn.ReLU()

    def forward(self, x): 

        x = self.relu0(self.norm0(self.conv0(x)))
        dilated1 = self.relu1(self.norm1(self.conv1(x))) 
        dilated2 = self.relu2(self.norm2(self.conv2(dilated1)))
        dilated3 = self.relu3(self.norm3(self.conv3(dilated2)))
        dilated4 = self.relu4(self.norm4(self.conv4(dilated3)))
        dilated5 = self.relu5(self.norm5(self.conv5(dilated4)))
        dilated6 = self.relu6(self.norm6(self.conv6(dilated5)))
		
        return (dilated1 + dilated2 + dilated3 + dilated4 + dilated5 + dilated6)


#Unet Model with (RGB Image input and 13 predicted classes of output)
class DeepDilatedUnet(nn.Module):
    def __init__(self, feature_scale = 8, in_channels=3, out_channels=1, features=32, n_class=13, dropout =0, nonlocal_mode='concatenation', attention_dsample=(1,1), is_batchnorm=True, is_2D_TORR = True, init_type = None ):
        super().__init__()

        self.is_batchnorm = is_batchnorm
        self.init_type = init_type
        self.nonlocal_mode = nonlocal_mode

        if is_2D_TORR and self.nonlocal_mode  == 'concatenation':
          self.nonlocal_mode = 'concatenation_mean_flow'

        filters = [64, 128, 256, 512]
        filters = [int(x / feature_scale) for x in filters]

        #Encoder
        self.encoder1 = ConvBlock(3, filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #NeckBottle
        self.neck1 = ParallelDilatedConvolutionModule(filters[2], filters[3], 3)
        self.gating = UnetGridGatingSignal2(filters[3], filters[2], kernel_size=(1, 1), is_batchnorm=self.is_batchnorm, init_type = self.init_type)

        # attention blocks
        if is_2D_TORR :

          self.attentionblock2 = GridAttentionBlock2D_TORR(in_channels=filters[1], gating_channels=filters[2],
                                                      inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode, 
                                                      use_W=False, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu',
                                                      init_type = self.init_type)
          self.attentionblock3 = GridAttentionBlock2D_TORR(in_channels=filters[2], gating_channels=filters[2],
                                                      inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode, 
                                                      use_W=False, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu',
                                                      init_type = self.init_type)

        else :

          self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2],
                                                      inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode,
                                                      init_type = self.init_type)
          self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[2],
                                                      inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode,
                                                      init_type = self.init_type)

        #Decoder
        self.upconv3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(filters[2]*2, filters[2])
        self.upconv2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
       	self.decoder2 = ConvBlock(filters[1]*2, filters[1])
        self.upconv1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(filters[0]*2, filters[0])
        self.drop1 = nn.Dropout(dropout)
        self.conv = nn.Conv2d(filters[0], n_class, 1)

        if self.init_type : 
          # initialise weights
          for m in self.modules():
              if isinstance(m, nn.Conv3d):
                  init_weights(m, init_type=self.init_type)
              elif isinstance(m, nn.BatchNorm3d):
                  init_weights(m, init_type=self.init_type)


    def forward(self, x):
        down1 = self.encoder1(x)
        down1pool = self.pool1(down1)
        down2 = self.encoder2(down1pool)
        down2pool = self.pool2(down2)
        down3 = self.encoder3(down2pool)
        down3pool = self.pool3(down3)
        neck1 = self.neck1(down3pool)

        # Gating Signal Generation
        gating = self.gating(neck1)

        # Attention Mechanism

        g_conv3, att3 = self.attentionblock3(down3, gating)
        g_conv2, att2 = self.attentionblock2(down2, gating)

        upc3 = self.upconv3(neck1)
        concat3 = torch.cat((g_conv3, upc3),dim=1)
        up3 = self.decoder3(concat3)

        upc2 = self.upconv2(up3)
        concat2 = torch.cat((g_conv2, upc2),dim=1)
        up2 = self.decoder2(concat2)

        upc1 = self.upconv1(up2)
        concat1 = torch.cat((down1, upc1),dim=1)
        up1 = self.drop1(self.decoder1(concat1))

        x = self.conv(up1)

        return x


