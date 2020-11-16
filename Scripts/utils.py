# Indoor Semantic Segmentation
#
# Date: 15/11/2020
# Authors: Xavier Riera, Albert Mestre, Jose Javier Gomez
# Institute: Universitat Politecnica De Catalunya
#

import numpy as np
import torch
import random
from matplotlib import pyplot as plt 


SEED = 1
random.seed(SEED)


#Define color dictionary to plot labels
#Link to visualize color vs meaning: 
#https://docs.google.com/spreadsheets/d/12LMGdwOMU_5dGKDGDEQaZWoj2xYNEL3FURyIB4R36s0/edit#gid=0
Convert_Label13_to_RGB = {
    0: np.array([0, 0, 1]),
    1: np.array([0.9137, 0.3490, 0.1882]),
    2: np.array([0, 0.8549, 0]),
    3: np.array([0.5843, 0, 0.9412]),
    4: np.array([0.8706, 0.9451, 0.0941]),
    5: np.array([1.0000, 0.8078, 0.8078]),
    6: np.array([0, 0.8784, 0.8980]),
    7: np.array([0.4157, 0.5333, 0.8000]),
    8: np.array([0.4588, 0.1137, 0.1608]),
    9: np.array([0.9412, 0.1373, 0.9216]),
    10: np.array([0, 0.6549, 0.6118]),
    11: np.array([0.9765, 0.5451, 0]),
    12: np.array([0.8824, 0.8980, 0.7608]),
    255: np.array([0, 0, 0]),
}

def array_colors_13(prediction, dictionary = Convert_Label13_to_RGB) :
  colors = np.array([])
  for x in prediction.numpy() :
    item = np.array([])
    for y in x:
        c = dictionary[y]
        item = c if (item.size==0) else np.vstack((item, c))
    colors =  np.expand_dims(item, axis=0) if (colors.size==0) else np.vstack((colors, np.expand_dims(item, axis=0)))
  return colors

#Class to store values of metrics and calculate efficiently average
class AverageMeter(object):
    #Inicialice the variables giving num
    def __init__(self, fmt=':f'):
        self.reset()

    #Method to put to 0 all values
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    #Method to introduces a new value and recalculate the average
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#Method to save the state of the model
import shutil
def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):

    filename = path + filename
    #save
    torch.save(state, filename)

    #copy if it is the best model yet
    if is_best:
      model_best_filename = path + 'model_best.pth.tar'
      #copy filename -> model_best_filename
      shutil.copyfile(filename, model_best_filename)


def load_checkpoint(path, filename='checkpoint.pth.tar'):
  filename = path + filename
  checkpoint = torch.load(filename)
  return checkpoint


#Method to put not labeled values
def add255Label(label, prediction):
  masks = np.not_equal(label,255)*prediction + np.equal(label,255)*255
  return masks


# Function plot_samples
#
# based on the work,
# Image Classification with a Multi-Layer Perceptron
# Notebook created in PyTorch by Santi Pascual for the UPC School (2019).

# https://colab.research.google.com/drive/1JlN3y_2cIE5pVR3Iyxjstri7Wosg5976
#
# Let's make a small function to show some example images. You can see the type of 3x640x480 images we are dealing with.

def plot_samples(images,M=2, N=5, title =""):
    # Randomly select NxN images and save them in ps
    ps = random.sample(range(0,images.shape[0]), M*N)
    # Allocates figure f divided in subplots contained in an NxN axarr
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html
    f,axarr = plt.subplots(N, M, figsize=(48,48))
    # Index for the images in ps to be plotted
    p = 0
    # Scan the MxN positions of the grid
    for i in range(N):
        for j in range(M):
          
            # Load the image pointed by p
            im = images[ps[p]].transpose()
            axarr[i,j].set_title(str(ps[p]))
            axarr[i,j].imshow(im)
            # Remove axis
            axarr[i,j].axis('off')
            # Point to the next image from the random selection
            p+=1
    # Show the plotted figure         
    f.suptitle(title, fontsize=40)
    plt.show()


#Make a toy sampleset without normalization for the sake of visualizing some samples

def toy_sampleset(images):

  sampleset = images
  img = []

  for n in range(500 if images.shape[0] >= 500 else images.shape[0]):
    img.append(torch.tensor(sampleset[n][None, ...].astype(np.uint8)))

  img = torch.cat(img, dim=0).data.numpy()
  return img
  



from torch.utils.data import DataLoader

NOT_LABELED_VALUE = 255
NOT_LABELED_COLOR = 13

# Function check_dataset
#

def check_dataset(dataloader, batch_size, array_colors, permute = False, stop = False):
 idx = 0

 for data in dataloader:
    images , labels = data  
    idx += 1
    if idx == 1:
      for i in range(batch_size):
        plt.imshow( images[i].permute(1, 2, 0) if permute else images[i] )
        plt.show()
        label = labels[i]
        label[label==NOT_LABELED_VALUE] = NOT_LABELED_COLOR
        plt.imshow(array_colors[ label ])
        plt.show()
    else :
      if stop : break   


# Function summary
#
def summary(path_exp, name_experiment, filename='model_best.pth.tar'):
  state = load_checkpoint(path_exp + name_experiment + '/', filename)
  best_iou        = state['iou']
  best_iou_dict   = state['iou_dict']
  best_epoch_acc  = state['acc']
  best_epoch      = state['epoch']
  print("-------------------------------------------------------------------------------------------------------------------")
  print("Summary: ")
  print("Best epoch = " + str(best_epoch) + ". IoU = " + str(best_iou) + ". Pixel Accuracy = " + str(best_epoch_acc) + "." )
  metrics.plot("IoU", best_iou_dict, class_name_13, color_class_13, number_of_classes = 13)
