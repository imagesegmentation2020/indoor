# Indoor Semantic Segmentation
#
# Date: 15/11/2020
# Authors: Xavier Riera, Albert Mestre, Jose Javier Gomez
# Institute: Universitat Politecnica De Catalunya
#


#torch
import torch

#numpy
import numpy as np

#Transforms
from torchvision import transforms

#Import Image library
from PIL import Image

#Import library to read CSV files
import pandas as pd

#Import library to get images
from skimage import io

#transformation 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


#Generate dataset using csv file 
class IndoorDataset(Dataset):

  def __init__(self, csv_file, root_dir, transform = None, transform_img = None):
    self.dataset = pd.read_csv(root_dir + csv_file)
    self.root_dir = root_dir
    self.transform = transform
    self.transform_img = transform_img

  def __len__(self):
      return len(self.dataset)

  def __getitem__(self,index):


    RGBImage = io.imread(self.root_dir + self.dataset.iloc[index, 0])                             
    labelsImage = io.imread(self.root_dir + self.dataset.iloc[index, 1])                      

    if self.transform is not None:
      RGBImage, labelsImage = self.transform((RGBImage, labelsImage))
    if self.transform_img is not None:
      RGBImage = self.transform_img(RGBImage)
    return RGBImage , labelsImage
	
