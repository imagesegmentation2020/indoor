# Indoor Semantic Segmentation
#
# Date: 15/11/2020
# Authors: Xavier Riera, Albert Mestre, Jose Javier Gomez
# Institute: Universitat Politecnica De Catalunya
#

import torch
import numpy as np
import matplotlib.pyplot as plt


#Metrics

#IOU

def calculate_iou(prediction, target, ignore_class=None):
	
	
	num_classes_target = np.unique(target)
	class_iou = {}
	for class_id in num_classes_target:
		if class_id != ignore_class:		
			pred_mask = (prediction == class_id).numpy()
			target_mask = (target == class_id).numpy()
			
			intersection = np.sum(np.logical_and(target_mask, pred_mask))
			union = np.sum(np.logical_or(target_mask, pred_mask))
		
			if (union == 0):
				iou = 0
			else:
				iou = intersection/union	 		

			class_iou[class_id] = iou

	return class_iou


#Method to plot table in tensorboard
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def IOUToTensorboard(validation_metric_epoch, train_metric_epoch, class_name_13, color_class_13):
  values_val = np.zeros(13)
  values_train = np.zeros(13)

  for (key,value) in validation_metric_epoch.items():
    values_val[key] = np.round(100*np.array(value).mean(),2)

  for (key,value) in train_metric_epoch.items():
    values_train[key] = np.round(100*np.array(value).mean(),2)

  A = [values_train, values_val]


  tableFigure = plt.figure(figsize=(14,2))
  table = plt.table(cellText=A,
                    colLabels = class_name_13,
                    colColours = color_class_13,
                    rowLabels = ['Train' , 'Validation'],
                    loc='center')
  ax = plt.gca()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  table.set_fontsize(14)
  plt.box(on=None)


  canvas = FigureCanvas(tableFigure)
  canvas.draw() 
  buf = canvas.buffer_rgba()
  X = np.asarray(buf)[:,:,:3]

  plt.close()

  return X


#Pixel Accuracy

def accuracy(prediction, target, ignore_class=None):
    """Computes the accuracy of predictions.
    Args:
        prediction (Tensor): The predictions.
        target (Tensor): The targets.
    :rtype: int
    """
    same = np.sum( (prediction == target).numpy() )
    num_pixels = target.numel()

    if ignore_class is None:
    	return same/num_elem
    else:
      ignored_pixels = np.sum( np.equal(target, ignore_class).numpy() )
      return (same - ignored_pixels) / (num_pixels - ignored_pixels)


## Metric averages

def average(lst): 
    return sum(lst) / len(lst) 

def average_simpledict(simpledict):
  list_values  = [value for (key, value) in simpledict.items() ]
  array_values = np.array(list_values).flatten()
  average = array_values.mean()
  return average

def average_complexdict(complexdict, number_of_classes = 13):
  list_values = [ np.array(value).mean() for (key, value) in complexdict.items() ]
  average = np.sum(list_values)/number_of_classes
  return average
  
  
## Metric Log
 
def log_metric_batch(data_name, simpledict, class_name, average, metric_name):
  print("----------------------------------------")
  print(data_name)
  print ("Batch metric " + metric_name + " results :  ")
  simpledict =  {class_name[key] : value for (key, value) in simpledict.items()}
  print(metric_name + '=', simpledict)
  print("Batch mean " + metric_name +" = " + str(average))


def log_metric_epoch(data_name, complexdict, class_name, average, metric_name):
  print("----------------------------------------")
  print(data_name)
  print ("Epoch metric " + metric_name + " results :  ")
  complexdict =  {class_name[key] : value for (key, value) in complexdict.items()}
  print(metric_name + '=', complexdict)
  for (key, value) in complexdict.items():
    print( "mean [" + key + "] : " + str(np.array(value).mean()) )

  print("Epoch mean " + metric_name + " = " + str(average))
  
## Plots
def plot_metric(data_name, simpledict, class_name, color_class, number_of_classes = 13):

  print("plot methric is deprecated. Use 'plot' instead")	
  plot(data_name, simpledict, class_name, color_class, number_of_classes)


def plot(data_name, simpledict, class_name, color_class, number_of_classes = 13):

  values = np.zeros(number_of_classes)

  for (key,value) in simpledict.items():
    values[key] = np.round(100*np.array(value).mean(),2)

  A = [values]

  tableFigure = plt.figure(figsize=(number_of_classes+1, 1))
  table = plt.table(cellText=A,
                    colLabels = class_name,
                    colColours = color_class,
                    rowLabels = [data_name],
                    loc='center')
  ax = plt.gca()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  table.set_fontsize(14)
  plt.box(on=None)
  
  
