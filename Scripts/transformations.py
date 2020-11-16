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

#random
import random


#Class Resize And RandomCrop for training
class ResizeAndRandomCrop(object):
    def __init__(self, size=300, crop=256):
        self.size = size
        self.crop = crop

    def random_crop(self, image, labels, crop):
      i, j, h, w = transforms.RandomCrop.get_params(image, [crop, crop])
      image = transforms.functional.crop(image, i, j, h, w)
      labels = transforms.functional.crop(labels, i, j, h, w)
      return image, labels

    def __call__(self, sample):

        image,target = sample

        #Resize image
        image = transforms.functional.resize(image, self.size)
        target = transforms.functional.resize(target, self.size, interpolation=Image.NEAREST)

        #randomCrop
        image, target = self.random_crop(image,target, self.crop)
        
        return (image, target)

#Class Resize and centercrop for validation and test
class ResizeAndCenterCrop(object):
    def __init__(self, size=300, crop=256):
        self.size = size
        self.transforms=transforms.Compose([transforms.CenterCrop(crop)])

    def __call__(self, sample):

        image, target = sample

        #Resize image
        image = transforms.functional.resize(image, self.size)
        target = transforms.functional.resize(target, self.size, interpolation=Image.NEAREST)

        #Centercrop
        image = self.transforms(image)
        target = self.transforms(target)

        return (image, target)

##Class RandomHorizontalFlip data augmentation
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample): 
      image, target = sample
      if random.random() < self.flip_prob:
          image = transforms.functional.hflip(image)
          target = transforms.functional.hflip(target)

      return image, target


#class to pass a bunch of transforms as parameter to apply to a tuple (image, label)
class ToTransforms(object):
    def __init__(self, transforms):
      self.transforms=transforms
    
    def __call__(self, sample):

        image, target = sample

        #Resize image
        image = self.transforms(image)
        target = self.transforms(target)

        return (image, target)


#Class convert to PIL image torchvision.transforms
class ToPILImage(object):
   
    def __call__(self, sample):

        image, target = sample

        #toPILImage
        image = Image.fromarray(image)
        target = Image.fromarray(target)

        return (image, target)



#Convert ndarrays in sample to Tensors.
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, target = sample

        #Convert to Tensor
        image = transforms.functional.to_tensor(np.array(image))
        target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return (image, target)

