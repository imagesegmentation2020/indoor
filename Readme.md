# Repository structure

The repository has the following structure:
  - Folder Scripts:
      - datas: class to read images from a CSV file
      - metrics: methods to calculate our metrics and show results.
      - transformations: classes and methods to be available the images and labels to be transformed
      - models: U-net model
      - utils: Other methods for miscellaneous goals, as save the current state of the model, select the colors to show labels.
  - Images: Zip file with all images and labels.
  - Model training: Google Colab to train the network.
  - Model testing: Google colab to load the model and test them.
  - Weights calculation: Google Colab to calculate the weights for the CrossEntropy weighted loss.
  - Tensorboard Unet Experiments: Google Colab to read results of previous executions.
  - Data preprocessing: Google colab to read the dataset and generate images and labels for training, test and validation.
  - Model Deeplabv3-Resnet101: Training code adapt to use a pretrained model.

# About

  - Date: 17/11/2020
  - Authors: Xavier Riera, José Javier Gómez, Albert Mestre
  - Institute: Universitat Politècnica de Catalunya.

# Motivation

We decided to pursue an indoor image segmentation task, which is one of the key problems in the field of computer vision.

One of the main reasons we chose this topic is because we are interested in object detection and semantic segmentation, and after analyze the datasets available for free, we found a dataset available for semantic segmantation.

The architectures normally used in this field also caught our attention.

We were excited to learn the most about deep learning, its implementation, and observe how far our creativity can go to improve the performance of our model.

Besides learning about Deep Learning, we seeked to learn the most about working as a team on a project of this type.

# Proposal

  - Analyze and pre-process the data adapting it to the network.
  - Learn how to code a semantic segmentation neural network from scratch, with the help of existing papers of U-Net.
  - Mitigate the class imbalance to achieve a better performance of the model.
  - Know about how the different loss functions work.
  - Learn to implement the metrics to quantify the performance of the model.
  - Introduce to methods to reduce the overfitting, for example, using data augmentation and regularization techniques.

# Milestones

  - Obtain the NYU Depth V2 Dataset and preprocess it.
  - Implement the functions to calculate the metrics and the loss to evaluate the performance of the model during training.
  - Write the semantic segmentation U-Net network from scratch.
  - Train the network considering the class imbalance, using the metrics to backpropagate and update the weights.
  - Improve the performance of the network by changing the hyperparameters.
  - Generate the report, presenting the results obtained with its conclusions. 

# Computational resources

In this project we used Google Colaboratory, which is a free platform where the users can write text and code, so it is executed and saved in the cloud. We made a Google account to store the results of the experiments.

Google colaboratory assigns a random GPU when running, which it sometimes has different computing power. It is a Nvidia GPU, which specifications are described in the following image. There are also daily time limitations of execution of the GPU.

![picture](https://drive.google.com/uc?export=view&id=1PI4PBjLLJdZvevrVy5wVl-ZJzpjSecIG)


# Dataset

The NYU depth V2 dataset [1] contains a wide variety of indoor scene images captured by both the RGB and Depth cameras. It features 1449 RGB images, aligned with its label images and Depth images:

![picture](https://drive.google.com/uc?export=view&id=1I5-XL0mhVsRo7F15uYHXnj2H9o6V-5Nq)


The depth images have not been used in this project. The raw images with its corresponding labelled images, which contain the ground truth of the object class for every pixel in the image, have been used to train, validate and test the network.

The size of raw images, which were captured by an RGB camera, is 640 x 480 x 3 channels. The size of the labelled images is 640 x 480 x 1 channel. Containing only the values of the object class for each one of the pixels.

This dataset had already defined the splitting of the dataset in training and test sets of images. The training and validation splits are generated randomly with proportionality 80% training and 20% validation using the given training split. The aim of that is to be able to compare the results and the performance of the network with other people who have used this dataset for the image segmentation task as well.

The splitting of the dataset is shown at the following table:

|            | Number of images | Percentage |
|------------|------------------|------------|
| Train      | 636              | 44%        |
| Test       | 654              | 45%        |
| Validation | 159              | 11%        |

### Interpretation of the dataset

By default, the objects labelled in the NYU Depth V2 dataset are classified in 894 different classes. To reduce the complexity of the project and achieve a better understanding of the dataset, a class reduction has been applied, grouping the 894 different classes in 13 classes, that contained more generalizing information of the objects. To grouping the classes a relation to convert 894 to 40 classes and 40 classes to 13 is given by the dataset. For example, the objects “shelf”, “wardrobe” and “desk” have been grouped into the class “furniture”.

In the following image we can observe the apparition of the different classes in the dataset, for train, validation, and test sets.

![picture](https://drive.google.com/uc?export=view&id=1nYneXdLBP60AiwfR3GGYTtcQqzvlRTsQ)

The labelled images in the dataset contain discrete values between 0 and 13, one for each object class, where the value “0” is assigned to those pixels which are not labeled.

We have adapted the labelled images values so they go between 0 and 12, and we also assigned to the “no label” pixels a value of 255 instead of 0.
 
Colorization of the labelled images has also been implemented in order to visualize the classes by looking the labelled image:

![picture](https://drive.google.com/uc?export=view&id=1fAIS2o9z2BuvhfUEIW8GRecKJfuDgAUa)

### Transformations applied to the dataset

A custom dataset class of the loading images and targets was created, in order to automatically resize them to fit them in the network, and also apply some data augmentation techniques.

Raw images have been resized to 300x300 pixels. Afterwards, we applied random crop to the train images and CenterCrop to validation and test images, so all the dataset have been transformed to 256x256 pixels.

This dataset class also applies some transformations like random cropping and center cropping used for data augmentation, controlling that the random is applied at same time for both the image and the target.

For data augmentation, we added some noise to the data and modified contrast, saturation and brightness, and also applied horizontal flip to the images, to try to improve the performance of the network and its generalization.

The values of the RGB images and its labels have been normalized to facilitate the training of the network, so all the values are between 0 and 1.

# Architecture

The architecture selected for our task is the U-Net, one of the most popular networks for computer vision, and specially for segmentation tasks.

This model was born in 2015 to solve image segmentation needs for biomedical applications [2]. It consist on a encoder, a bottleneck and a decoder and would be detailed at the paragraph “Disclosing the U-net”
![picture](https://drive.google.com/uc?export=view&id=1RdO40r0BoX8TBFL4K62stI4i14zOjg2s)

Our model consists of using the U-Net to convert 256x256 RGB images to get a 256x256 pixels segmentation map of 13 categories. To accomplish that we follow the same structure, but adapt the steps of the original model to our needs. (Mainly modifying the convolutional part).
![picture](https://drive.google.com/uc?export=view&id=1fwa_Iws6h0nkdpuRrIHK3Eal8EnmJ_b8)

The main difference between U-net and other network architectures is that every pooling layer is mirrored by an up-sampling layer.
The mirror permits to reduce the size of the image in the encoder part, reducing the number of parameters that should be calculated.
For the previous reason, and also as the number of filters can be parallelized using GPUs is it possible to train the network fastly.
 
### Disclosing the U-NET

It is possible to define the U-net as a convolutional Encoder-Decoder architecture because it is made by an encoder, a decoder and a Neckbottle.

**Encoder**

The encoder part has the goal to obtain information about the input image. It is made by four convolutional blocks and four maxpooling steps.
Every Convolutional block is constructed by two blocks of Conv2D, BatchNormalization and ReLU as we can see on the following image:

![picture](https://drive.google.com/uc?export=view&id=11e5bPDEelHbZ-5oQWfqKQCLgB5BzaEnn)

The different elements of the Conv2D has the following goals:
  - Conv2D: Get information of the input image using different kind of filters.
  - Batch Normalization: Block to modify the layer, re-centering and re-scaling them to made faster and more stable.
  - ReLU: Discard negative values.

The maxpooling select the highest value of every 2x2 cell at the end of every convolutional block. The goal of maxpooling is to retain only the important features from each region and throw away non relevant information.

**Bottleneck**

The bottleneck block has the goal to force the model to learn a compression of the input data. The idea is that this model only learns relevant information to try to reconstruct the image.
The bottleneck is made by a convolutional block. It can also be used to represent the inputs with reduced dimensionality, because it contains all the information of the input image.

**Decoder**

The goal of the decoder is to predict the labels of the input image, using the information obtained by the Encoder and Bottleneck part.
The decoder is made by four unpooling blocks, four convolutional blocks, an optional dropout layer and finally a 2Dconvolution.

The elements of the Decoder part have the following goals:
  - Unpooling: Increase the size of the features, to try to recover a label with the same dimensions of the input image.
  - ConvBlock: Use the unpooling features and the higher resolution feature maps from the encoder network to reconstruct features. The feature maps from the encoder network help the decoder part generate reconstructions with accurate details.
  - Dropout: Ignore certain information to try to reduce overfitting problems.
  - Conv2D: Assign a pseudo probability for every input pixel to pertain to a certain class.

The output of the Conv2D of the last layer should have the same number of features of the number of classes that should predict.

# Loss functions

The loss function has the goal to measure how far the prediction of the network is with respect to the expected results.
The Cross Entropy is the simplest and most common loss function used in semantic segmentation. This loss examines each pixel individually, comparing the class prediction to our one-hot encoded target vector.

Mathematically is calculated using the following formule:

![picture](https://drive.google.com/uc?export=view&id=1Gebq9XjYP-vYP9KBvH0RFoWqEWADW1gC)
![picture](https://drive.google.com/uc?export=view&id=1EeVWPRR6fHczZCpAy_02TKGSHcN5IX7P)  is the target vector.
![picture](https://drive.google.com/uc?export=view&id=1eL1ESdVTn4o2K3JfDk-K_PaUefU-bg4r) is the softmax function calculated as:
![picture](https://drive.google.com/uc?export=view&id=1Kyl4WvPQNOreGduZF9oPYziDTNsGMkyM)
![picture](https://drive.google.com/uc?export=view&id=1nkSw43m9lmXOFA_ND4WEi5dzsyFladt-) is the CNN score of the class 'i'

Due to the pixels that are not labeled in the target image can’t be predicted, they are not considered in this formula.

One of the problems of the Cross Entropy loss is the class imbalance of the dataset, because it learns better classes that appear more than classes that appear less.

To reduce this effect, it is possible use the "Weighted Cross Entropy" that consists of considering more relevant pixels that appear less. This relevance should have an inverse proportion with the number of pixels. For this reason it is necessary to analyze the number of pixels of each class in the training dataset:

![picture](https://drive.google.com/uc?export=view&id=1iTEIs-XGHr8Q9GBRoLIeO1SOS140j3dI)

To calculate the weights that should be used in the weighted loss we need to calculate the inverted frequency. These are the weights for our train dataset.

![picture](https://drive.google.com/uc?export=view&id=1LWw1xp_9XEjxtrJZZp4218WtEIHWGf7u)

With this loss ideally it is possible to increase the mean intersection over union, but as it is explained on the experiments, it won’t improve them because the weights of the training are not the weights that maximizes the results of the validation dataset. One of the reasons is because of the size of the dataset, which is not so big.
Each weight is calculated as:

![picture](https://drive.google.com/uc?export=view&id=1JK0b6PjAI8QdYtYmhnvGMenJrF-Aj8Jy)

Where 13 is the number of classses

![picture](https://drive.google.com/uc?export=view&id=11-aVaFoS3Med92m80ezNQYNvFKbk76uq)

![picture](https://drive.google.com/uc?export=view&id=1m1l_-frWn0MJzdcDO2Et_1ihFYIeH8rs) is the number of pixels of class 'i'

These weights are used by the Cross Entropy loss to multiply the result by the correspondent factor.

# Metrics

Metrics are used in semantic segmentation to evaluate results of the training of the network. In our application the goal is maximize the metric called mean intersection over union.
By using the metrics, parameters of the model can be selected to maximize the needs of the application.
To evaluate the metrics, pixels that are not labeled on the ground truth are not considered.
The procedure to calculate the metrics in every epoch are the following one:
  - Every batch the values of pixel accuracy per class and intersection over union per class are calculated and stored in a dictionary.
  - Every epoch
     - Pixel accuracy per class is calculated averaging the pixel accuracy per class of every batch.
     - Mean pixel accuracy is calculated averaging the pixel accuracy per class of every epoch.
     - Intersection over union per class is calculated averaging the intersection over union per class of every batch.
     - Mean intersection over union is calculated averaging the intersection over union per class of every epoch.

**Pixel Accuracy per class**

Consist on evaluating for each class the percentage of pixels that are well labeled.
The pixel accuracy per class and per batch is calculated as:

![picture](https://drive.google.com/uc?export=view&id=14JpxsBSdwwqoP9z_s9b3KTs2adKjLKsH)

![picture](https://drive.google.com/uc?export=view&id=1nA9hiG9-ma70kbsLk07dX03nWjs2ZcCx) is the pixel accuracy on batch 'b' of class 'i'

![picture](https://drive.google.com/uc?export=view&id=18YrD3zVY6DzkpUj9H5RoDNxFl0JB-nqC) is the number of pixels of class 'i' of the batch 'b' well labeled

![picture](https://drive.google.com/uc?export=view&id=1oesgMzZ9gae9g4zhHHqvRu-PEfy2PSVr) is the number of pixels of class 'i' on batch 'b'

The pixel accuracy per class is calculated using the following formula:

![picture](https://drive.google.com/uc?export=view&id=1g1rkhg0fMYJFziqnxWn39AKYOdlNY4hU)

![picture](https://drive.google.com/uc?export=view&id=1aZC_1q8c2ULYgZDBRp5HlKtIoaTHjGNf) is the pixel accuracy of class 'i'

![picture](https://drive.google.com/uc?export=view&id=10_ZOPMb2TwRMRxmjdXPRkxAzYH3AC69y) is the number of batch that contains the class 'i'

**Mean pixel accuracy**

The mean pixel accuracy, as his name defines, compute the mean of the pixel accuracy of the classes calculated in the previous paragraph.
The formula used to calculate them is the next one:
![picture](https://drive.google.com/uc?export=view&id=1cE6SOywr7bqTOtclKxFXAd9gV0MFWxft)

**Intersection over union per class**

The intersection over union per class is calculated as the relation between the pixels well labeled of the class divided by the number of pixels predicted as the class and the number of pixels of the class that are not predicted.

Graphically is calculated as:

![picture](https://drive.google.com/uc?export=view&id=12A2zeNEumA0EZ7FQJcaAV-TMrg-BzJb_)

The intersection over union per class and per batch is calculated as:

![picture](https://drive.google.com/uc?export=view&id=1Ro_pi5naGR4R6yhMtTHop-M4OwDSHuUG)

![picture](https://drive.google.com/uc?export=view&id=1VTWaNb640zxlVlAbHWZ8A_as4B81DnZz) is the intersection over union on batch 'b' of class 'i'.

![picture](https://drive.google.com/uc?export=view&id=18YrD3zVY6DzkpUj9H5RoDNxFl0JB-nqC) is the number of pixels of class 'i' of the batch 'b' well labeled.

![picture](https://drive.google.com/uc?export=view&id=1MoQn3_L13aN7Rptob_YG7L6qOyzPwUpG) is the number of pixels of class 'i' of the batch 'b' labeled as class 'i' but pertaining to other class.

![picture](https://drive.google.com/uc?export=view&id=1UFmmL_s4GcNI4FxiekPhOuCk-L-Dxa1s) is the number of pixels of class 'i' of the batch 'b' labeled as other class but pertaining to class 'i'.

The pixel accuracy per class is calculated using the following formula:

![picture](https://drive.google.com/uc?export=view&id=1oluhj4nL2hk_kdz1Ny0FO2ibIsV8XqAh)

![picture](https://drive.google.com/uc?export=view&id=1ORS7CoEE6G9LFVXz5jyFOEJ1OFWMv3uh) is the intersection over union of class 'i'.

![picture](https://drive.google.com/uc?export=view&id=10_ZOPMb2TwRMRxmjdXPRkxAzYH3AC69y) is the number of batch that contains the class 'i'

**Mean intersection over union**

The mean intersection over union, compute the mean of the intersection over union of the classes calculated in the previous paragraph.

This metric helps to reduce the effect of the class imbalance, for the following reasons:
  - Mean pixel accuracy doesn't consider the false positives, and therefore when the network has doubts about predicting the class tends to select one of the classes that appears the most.
  - In mean intersection over union, the classes with less apparitions have the same influence in the metric than classes with higher apparitions, due to the nature of the calculation of this metric. This makes it a good metric to evaluate unbalanced datasets, as is the case of the dataset in this project.

The formula used to calculate them is the next one:
![picture](https://drive.google.com/uc?export=view&id=1Gze9clV00RvmGlqxKNuuyt1Vnow3oSPo)

# Experiments

We trained the model several times with different optimizers and hyperparameters, aiming to get the best Mean IOU. Which is the metric that best defines the performance of the results, as it is explained above. 

The main criterion to select which model is the best for our application is based on the simulation that gets the best mean IOU in any epoch of its simulation. Other factors can be taken into account, for example theoretical reasons or hardware limitations.

In every experiment the mean IOU, the Pixel Accuracy and the loss are graphically evaluated.

**Experiment 1: Adam learning Rate comparison**

The first hyperparameter evaluated has been the learning rate. The same optimizer has been used in both experiments, the Adam Optimizer.
The ADAM optimizer was selected to our model due to his adaptive gradient and his computacional efficiency.

In this experiment two of the typical LR for ADAM are compared:

  - Learning rate 1e-3
    - ![#f03c15](https://via.placeholder.com/15/ff8000/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/3B83BD/000000?text=+) Validation 
  - Learning rate 5e-4
    - ![#f03c15](https://via.placeholder.com/15/B22222/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Validation 

Before the experiment theoretically we expect slightly better results in 5e-4 learning rate.

![picture](https://drive.google.com/uc?export=view&id=1rypLPBpnUxM1b5Wch5FONKNctmoeSgBm)![picture](https://drive.google.com/uc?export=view&id=1gLrfPZEBVauqxjcnSXO5o5CVDVAucM7m)![picture](https://drive.google.com/uc?export=view&id=12qHHkz8Jo8SFzh_dyPl1apV-aQHNwQc0)

|                | ADAM LR 1e-3 | ADAM LR 5e-4 |
|----------------|--------------|--------------|
| Best epoch     | 165          | 191          |
| Mean IOU       | 41.09%       | 41.64%       |
| Mean Pixel Acc | 65.73%       | 67.27%       |

We observe that the one with learning rate 1e-3 converges faster, but slightly better results in metrics are obtained with learning rate 5e-4. 

**Experiment 2: Weight decay regularization**

The second parameter to evaluate is weight decay regularization.
Using the learning rate with the best IOU of the previous step (5e-4), we are going to simulate the difference between using Weight decay 1e-4 and not using it. 

  - Without using regularization
    - ![#f03c15](https://via.placeholder.com/15/B22222/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Validation 
  - Using weight decay 1e-4
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/C2185B/000000?text=+) Validation 

Before the experiment theoretically we expect reduce the gap between train and validation curves using regularization.

![picture](https://drive.google.com/uc?export=view&id=1ScsH5cXHZT7psDjwGQDtdIlqD4wly9vs)![picture](https://drive.google.com/uc?export=view&id=1QUWvZgZ9a2Rcf4JF-n_c4eCj3rc886CQ)![picture](https://drive.google.com/uc?export=view&id=1IaBiQPXtNdWW8H0CvvboYksYXs97r-fh)

|                | ADAM Decay 0 | ADAM Decay 1e-4 |
|----------------|--------------|-----------------|
| Best epoch     | 191          | 216             |
| Mean IOU       | 41.64%       | 42.54%          |
| Mean Pixel Acc | 67.27%       | 67.50%          |

The  regularization does not reduce the overfitting effect but improves slightly the results obtaining an increment of IOU about 1%

**Experiment 3: Comparison between Optimizers**

The third experiment consists of evaluating a train between the best optimizer in the previous step (ADAM with Learning rate 5e-4 and weight decay 1e-4) with a SGD with momentum with similar conditions.

  - ADAM
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/C2185B/000000?text=+) Validation 
  - SGD with momentum
    - ![#f03c15](https://via.placeholder.com/15/ff8000/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/3B83BD/000000?text=+) Validation 

Before the experiment theoretically we expect SGD with momentum has similar properties than ADAM but trains slower.

![picture](https://drive.google.com/uc?export=view&id=1PZ2Xs9uW7P3dw5CEeYZ-m0qMzvZcQCG9)![picture](https://drive.google.com/uc?export=view&id=1QZ3PM28o3w6uqWQTsm-lzbLV2uYctyjd)![picture](https://drive.google.com/uc?export=view&id=1aQ1flBGHnMXp6Lwt6uxM31ksyVFUvQWo)

|                | ADAM Decay 0 | SGD with Momentum |
|----------------|--------------|-------------------|
| Best epoch     | 216          | 379               |
| Mean IOU       | 42.54%       | 30.60%            |
| Mean Pixel Acc | 67.50%       | 58.90%            |

Visualizing the graphics is easy to select ADAM as the best optimizer for the model. ADAM Optimizer is faster than SGD with Momentum, and due to this reason, when the system is trained with a limited number of epochs ADAM get better results.
For this reason, because the number of epochs due to restrictions of hardware and time to evaluate, we selected the ADAM optimizer although the IOU of the SGD continues increasing at the last epoch simulated.

**Experiment 4: Comparison between Dropouts**

The fourth experiment consists of trying to reduce the overfitting, putting a dropout stage before the last convolutional layer.

  - No Dropout
    - ![#f03c15](https://via.placeholder.com/15/3B83BD/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/B22222/000000?text=+) Validation 
  - 0.2 Dropout
    - ![#f03c15](https://via.placeholder.com/15/ff8000/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/CDCDCD/000000?text=+) Validation 
  - 0.5 Dropout
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/C2185B/000000?text=+) Validation 

Before the experiment we expect dropout helps to reduce overfitting and therefore improve results with dropout.

![picture](https://drive.google.com/uc?export=view&id=1TUZvPhwcsI5nfhA3N7TKXCVyTYIt_Ez0)![picture](https://drive.google.com/uc?export=view&id=1-nyiakPO7dTLi1PyZQ5z3K19vDkKvmqP)![picture](https://drive.google.com/uc?export=view&id=1age_7vnSTxspjYveayPTUl-_lGErJsqw)

|                | No dropout | Dropout 0.2 | Dropout 0.5 |
|----------------|------------|-------------|-------------|
| Best epoch     | 216        | 354         | 421         |
| Mean IOU       | 42.54%     | 42.59%      | 41.34%      |
| Mean Pixel Acc | 67.50%     | 68.58%      | 68.60%      |

A dropout with a probability p=0.5 (50%) does not increase the metric performance.
Instead, a dropout with a probability p=0.2 (20%) seems to get similar metric performance.
According to deep learning theory, the experiments with dropout should provide better results. Even though the results are very similar, we obtained slightly better results with dropout 0,2, so we kept this dropout in the following experiments.

**Experiment 5: Changing the loss function to try to reduce class imbalance**

As it is explained on paragraph "interpretation of the dataset" our dataset is imbalanced
During training, it is possible to appreciate that using standard CrossEntropy, classes that appear more in the dataset are learnt better than classes that appear less.
For this reason the following experiment consists of force to the system to attend more to classes that appear less on the images.
To do this experiment we are going to use the weighted loss explained in the $$"Loss functions"$$ paragraph

  - Standard Cross Entropy
    - ![#f03c15](https://via.placeholder.com/15/B22222/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Validation 
  - Weighted Cross Entropy
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/C2185B/000000?text=+) Validation 

Previous at the experiment we expect improve results on intersection over union, and reduce also the class imbalance, improving the results of classes with less apparitions on the dataset.

![picture](https://drive.google.com/uc?export=view&id=1i5aOdOTZCqqAm6BpEKD4HU4jpOTDdkrK)![picture](https://drive.google.com/uc?export=view&id=1ud45Yn9QnKnMmcOvZWLmUHmUBHX1ML1k)![picture](https://drive.google.com/uc?export=view&id=1WnILgAYBlM_z9yDB0gkubwkt_DTUhhKa)

|                | Standard Crossentropy | Weighted Crossentropy |
|----------------|-----------------------|-----------------------|
| Best epoch     | 354                   | 302                   |
| Mean IOU       | 42.59%                | 40.99%                |
| Mean Pixel Acc | 68.58%                | 66.87%                |

The weighted loss does not improve the mean IOU. The explanation to that fact may improve the results in the ideal case that the distribution of categories in the training split were similar to the distribution of categories in the validation split.

The results of the IOU per class are the following:
  - Standard Cross Entropy

![picture](https://drive.google.com/uc?export=view&id=1nxYPIaNwjpArDBRXx5vJ3O91SGh8pjvH)
  - Weighted Cross Entropy

![picture](https://drive.google.com/uc?export=view&id=1hNXyoNoit9G6zkim8ENmz-B9RmVEueMS)
Despite the mean IOU is not better than not using weighted loss, we can see that the classes with less appearance have increased their IOU, and the classes more relevant have decreased.

**Experiment 6: Add noise**

To try to make the system more robust to different brightness levels, we added different random brightness, contrast and saturation.

To do that it is possible using a function called ColorJitter, that consist on add noise, modifying brightness, contrast and saturation randomly.

  - Without ColorJitter
    - ![#f03c15](https://via.placeholder.com/15/B22222/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/00AAE4/000000?text=+) Validation 
  - With ColorJitter
    - ![#f03c15](https://via.placeholder.com/15/C2185B/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/009C8C/000000?text=+) Validation 

 We expect improve the results adding random brightness noise to make the system more robust with different kind of noisy images.

![picture](https://drive.google.com/uc?export=view&id=13kMIqCJE26sER68_d_Luaa7upE_jsmwf)![picture](https://drive.google.com/uc?export=view&id=1534P-EReYfgKIh0JOtsanYjc2_hE0p73)![picture](https://drive.google.com/uc?export=view&id=1SXGBn-JOe3DioeBJtVzS10azw1G7sHoV)

|                | Without ColorJitter | With ColorJitter |
|----------------|---------------------|------------------|
| Best epoch     | 354                 | 348              |
| Mean IOU       | 42.59%              | 41.61%           |
| Mean Pixel Acc | 68.58%              | 68.70%           |

After the results, we can see as a conclusión that adding noise in our dataset doesn't improve the results, probably because the pictures are in similar brightness conditions 

Using colorJitter for data augmentation, stabilizes the loss in the validation split, but the results obtained in the metrics are worse.

**Experiment 7: Horizontal flip**

In the following experiment we are going to compare the results using horizontal flip or without using them.

  - Without Horizontal Flip
    - ![#f03c15](https://via.placeholder.com/15/CDCDCD/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/ff8000/000000?text=+) Validation 
  - With Horizontal Flip
    - ![#f03c15](https://via.placeholder.com/15/B22222/000000?text=+) Train
    - ![#f03c15](https://via.placeholder.com/15/0E76A8/000000?text=+) Validation 

Interpretative to humans, we can detect the same objects if we have an image or if we have them with an HorizontalFlip.
For this reason it is reasonable that we can do a kind of data augmentation doing horizontal flip on training to try to decrease the overfitting.

![picture](https://drive.google.com/uc?export=view&id=1Vene1h7d6wczrqmYiPPVXS2c0SNWgfxs)![picture](https://drive.google.com/uc?export=view&id=1Qhu0siqj-fCaTmV4jE-lDTrVxGHVWohY)![picture](https://drive.google.com/uc?export=view&id=1GTsA-_OQ1PWmm6RflYuwHGIXdsBL2p29)

|                | Without Horizontal Flip | With Horizontal Flip |
|----------------|-------------------------|----------------------|
| Best epoch     | 354                     | 414                  |
| Mean IOU       | 42.59%                  | 43.56%               |
| Mean Pixel Acc | 68.58%                  | 69.69%               |

As we expected, the results on validation are better using Horizontal Flip and also the loss and the metrics are stabilized.

# Conclusions

After select the best model as the best mean intersection over union of all experiments, we can see the following results in testing:

  - Mean IOU: 34.09%
  - Mean Pixel Accuracy: 62.76%

![picture](https://drive.google.com/uc?export=view&id=1XFrcMIYJ9bnqiAEWan19qpDJxQL4nUMb)

Example of qualitative results:
  - Input image

![picture](https://drive.google.com/uc?export=view&id=1hIN0zWg5C5sVr2z7F3A_8C9GVCBwMiog)
  - Ground through
  
![picture](https://drive.google.com/uc?export=view&id=1bZcuA8TUwdvWfYnVUd_AK2zuoE1H7fL1)
  - Prediction

![picture](https://drive.google.com/uc?export=view&id=1GC_3gnkXJDyxWwMSBfL6eNsSup84JGub)

At the end of all of the project, all of the experiments and the qualitative results, we learn the following performances about deep learning:
  - Is posible train a net without invest money on GPU, thanks to Google Colab.
  - Data is all, having good results depends on the amount of data and its treatment.
  - Most of the time to program a network is to process the data and wait for training results.
  - Know metrics goals before start the net helps to modelate the parameters of the network.
  - Dropout: better results but less improves than we expect.
  - Intuitively the model learns better classes that are easy to differentiate for humans 
  - Human logic helps to select transformations to our system.

# References
[1]: NYU depth V2 dataset. https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

[2]: Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation". CVPR, 2015. https://arxiv.org/abs/1505.04597

[3]: Mapping to 40/13 class configuration. https://cs.nyu.edu/~deigen/dnl/

[4]: SceneNetv1.0. https://github.com/ankurhanda/SceneNetv1.0/

[5]: Train-test splits. https://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html 

[6]: scikit-image. https://scikit-image.org/docs/dev/api/skimage.io.html

[7]: Semantic segmentation reference training scripts. https://github.com/pytorch/vision/tree/master/references/segmentation

[8]: Train Imagenet, AverageMeter. https://github.com/pytorch/examples/tree/master/imagenet

[9] Implement the U-Net network from scratch. https://arxiv.org/pdf/1505.04597.pdf

[10] Intersection Over Union. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html

[11] Dice Loss. https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.dice_loss

[12] Image augmentation in machine learning experiments. https://imgaug.readthedocs.io/en/latest/

[13] Class-Balanced Loss Based on Effective Number of Samples. https://arxiv.org/pdf/1901.05555.pdf

[14] Semantic segmentation. https://neptune.ai/blog/image-segmentation-in-2020

[15] Unet. https://neurohive.io/en/popular-networks/u-net/

[16] Unet. https://arxiv.org/pdf/1505.04597.pdf

[17] Unet. https://github.com/jakeret/tf_unet/blob/master/tf_unet/unet.py

[18] Deeplab, Deeplab V3 Network. https://sthalles.github.io/deep_segmentation_network/

[19] Deeplab, Deeplab V3 Network. https://github.com/sthalles/deeplab_v3

[20] Deeplab, Deeplab V3 Network. https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74

[21] Mask_RCNN Network. https://github.com/matterport/Mask_RCNN

[22] Mask_RCNN Network. https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/

[23] Comparision between arquitectures. https://devblogs.microsoft.com/cse/2018/07/05/satellite-images-segmentation-sustainable-farming/

[24] Comparision between arquitectures. https://www.quora.com/Is-DeepLab-v3-better-than-Mask-R-CNN

[25] Dataset search by google. https://datasetsearch.research.google.com/

[26] Intersection-over-union and pixel accuracy. https://www.jeremyjordan.me/evaluating-image-segmentation-models/

[27] Intersection-over-union and pixel accuracy. https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

[28] U-Net and pixel accuracy results in class imbalance. https://towardsdatascience.com/image-segmentation-kaggle-experience-9a41cb8924f0






