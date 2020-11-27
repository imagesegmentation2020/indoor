# Indoor Semantic Segmentation
#
# Date: 15/11/2020
# Authors: Xavier Riera, Albert Mestre, Jose Javier Gomez
# Institute: Universitat Politecnica De Catalunya
#
#attention.unet_grid_gating_signal
#https://github.com/ozan-oktay/Attention-Gated-Networks


import torch
import torch.nn as nn
import torch.nn.functional as F
from networks_other import init_weights



class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1), is_batchnorm=True, init_type = None):
        super(UnetGridGatingSignal2, self).__init__()

        self.init_type = init_type

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        if self.init_type :
          # initialise the blocks
          for m in self.children():
              init_weights(m, init_type=self.init_type)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
		
		
class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
