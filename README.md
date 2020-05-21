# stvNet

## Overview

The goal of this project was to implement the 6D Pose Estimation solution described in the [PVNet](https://arxiv.org/abs/1812.11788) paper.

The main code is contained in 3 python files. ```models.py``` contains the neural net models, implemented using tensorflow and keras, as well as code for training models and saving and displaying model performance metrics. The ```data.py``` file contains the code used to convert the data provided in the LINEMOD dataset, to the actual objects used as target data for the neural networks. The file also contains the generator functions used to serve the training data. The ```pipeLine.py``` file implements the algorithm as described in the PVNet paper.

In brief, an RGB image is given as in input to the neural net model, which performs image segmentation, and also outputs pixel-wise unit vector predictions for each keypoint given in the training data. By examining the object pixels determined by the image segmentation output, we can take the associated unit vector predictions and perform a [RanSaC](https://en.wikipedia.org/wiki/Random_sample_consensus) voting process to determine a list of hypotheses weighted according to how well they fit with each of the other object pixels. The weighted average is then taken to give a value for each keypoint. These 2D keypoints, along with a set of corresponding keypoints in a 3D coordinate system, can then be passed to a [PnP solver function](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp) which gives the camera translation and rotation vectors in the object coordinate system.

We can measure the accuracy of the process by then using the [cv2 projectPoints](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints) function to plot the keypoints of 3D object according to our rotation and translation vectors. This generated set of keypoints can then be measured against the ground truth values to generate error metrics over a set of predictions.

I created a [youtube series](https://www.youtube.com/playlist?list=PL3om9a5CvNUl-ZUvZLS8z66uc0qOxIEqj) which goes into the design process for each of these files, and also describes the final results of the algorithm, along with the performance effects of models used, an alternate 3d label set, pruning hypotheses predictions, and the number of keypoint hypotheses considered. Also included in the repo is a ```demo.ipynb``` python notebook, which contains a demonstration of the pipeline process from the input, to the final prediction and error metrics, with a focus on the data objects used throughout the process.

## Instructions

### Custom Model

To build a custom model, construct the model within a function in the ```models.py``` file. The output of the model must match the models function (e.g. 18 outputs for vector prediction, 1 output for class prediction, or a combined output)

### Model Training

To train a model, add an entry to the ```modelsDict``` variable in ```models.py```. The key should be the model name, and the value should be a ```modelDictValObject``` which acts as a dictionary containing all the necessary details to train the model such as the appropriate training generator, loss functions, output shape, epochs, learning rate, metrics, alternate labelling, and augmentation.

Then, add the model name as the argument of a ```modelSet``` class into the ```modelSets``` list variable within the ```__main__``` block of ```models.py```. Then call ```	trainModels(modelSets)``` to train each of the models included in the ```modelSets``` variable according to their specified parameters.

### Model Testing and Results

Trained models can be evaluated using the default validation dataset. Setup the models that you wish to evaluate within the ```modelSets``` variable using the same process described in the Model Training section, and call ```evaluateModels(modelSets)```.

Plots displaying the loss history for each trained object can be generated using the plotHistories function. Setup the ```modelSets``` variable with the desired models and call ```plotHistories(modelSets)``` to generate graphs displaying the training and validation loss values over the models trained epochs.

### PipeLine

To see how a given model performs in the pipeline process, edit the ```pipeLine.py``` file and setup the ```modelSets``` variable with the desired models and call the ```evalModels``` function with the desired parameters.

If the ```saveAccuracy``` boolean was set to ```True``` in ```evalModels```, the accuracy metrics will be saved for the selected models and can be displayed by calling ```accuracyPlot(modelSets)```.

### Data Format and Processing

The neural net model is trained to detect either the 2d coordinates of a set of 3d object keypoints on an image, the pixels associated with an object of interest, or both. The functions used to generate the target data for the neural nets is found in the `data.py` file. (`coordsTrainingGenerator`, `classTrainingGenerator`, `combinedTrainingGenerator`)

These functions read data from the [LINEMOD](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EXK2K0B-QrNPi8MYLDFHdB8BQm9cWTxRGV9dQgauczkVYQ?e=beftUz) dataset, one of several datasets used in academic works in 6d pose estimation. I show the folder and talk about the data format in [this](https://www.youtube.com/watch?v=wbTdqlBXOOE) video in the youtube series, but did not include the folder in this repo due to size constraints. The dataset contains a folder for each object of interest, and within that folder, there is a `JPEGImages` folder, a `labels` folder, and a `mask` folder. `JPEGImages` contains the RGB images which are converted to numpy arrays and used as the input data for the neural net.

The mask folder contains a corresponding set of images that are made up of black pixels for pixels not associated with the object of interest, or white pixels for pixels associated with the object of interest. A (HxWx1) array is generated indicating whether a pixel belongs to the object of interest, which is used as target data for the class and combined generators.

The labels folder contains a corresponding `.txt` file for each image in `JPEGImages`, which gives information on the pixel location of the 9 bounding box keypoints. The format of these files is `{object classification tag} {x1} {y1} {x2} {y2} ... {x9} {y9}` the object classification tag denotes the object associated with the coordinates. Each of the x or y values is a value between 0 and 1 that gives the relative coordinate on the image (e.x. if x1 and y1 are .1 and .5 on a 640x480 image, the keypoint is located at pixel (64, 240)). In all generators we generate a `modelMask` array, and for each pixel belonging to the object of interest, we calculate a unit vector from the pixel to each 2d keypoint. The end result is a (HxWx18) array, containing a set of 9 unit vectors for each pixel that belongs to the object.

This project structure could be used on a custom object object given a dataset of the same format for the object. That is: a set of photos of the object of interest, a corresponding set of object masks of those photos identifying which pixels belong to the object, and a corresponding set of .txt files which give the 2d pixel locations of a set of 3d keypoints. A file containing the corresponding keypoint locations in a 3d cooridinate system should also be included for the PNP function.
