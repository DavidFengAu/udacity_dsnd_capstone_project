[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./app/ScreenShot2.png 

## Project Overview

Welcome to this project. It implements Convolutional Neural Networks (CNN) to identify an estimate 
of the dog’s breed from real-world and user-supplied images. The model was training by 8,351 dog images and can
classify dog into 133 categories.

In addition to predict canine’s breed, it also detect whether or not the user-supplied image including a dog. If not, it
will pop a message for non-dog image detected. If yes, it will tell the estimated breed.

![Sample Output][image1]

## Problem Statement
   
This project addresses the problem of identifying dog breed from images, and the result should not be effected by 
dog size, skin color and head-facing. The model should only return one prediction for each image.


## Metrics

The accuracy to identify whether or not an image including a dog should be over 95%.
The accuracy to identify the correct dog breed should be over 80%

# Data Exploration and Visualization

Training dog images are supplied by Udacity, and are downloadable 
from [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

There are 133 total dog categories and 8351 total dog images.

## Data Preprocessing

At the beginning, dog images are divided into training, validation and testing dataset. 
There are 6680 training dog images, 835 validation dog images and 836 testing dog images.

As I use TensorFlow as backend, Keras CNNs require a 4D array as input, 
with shape of (nb_samples, rows, columns, channels), 
where nb_samples corresponds to the total number of images (or samples), 
and rows, columns, and channels correspond to 
the number of rows, columns, and channels for each image, respectively.

The `path_to_tensor` function preprocess image to be ready for training. 
It takes the path of an image as input and returns a 4D tensor suitable for supplying to a Keras CNN. 
The function first loads the image and resizes it to a square image that is  224×224 pixels. 
Next, the image is converted to an array, which is then resized to a 4D tensor. 
In this case, since we are working with color images, each image has three channels. 
Therefore, the returned tensor will always have shape of (1, 224, 224, 3).

## Implementation

### Project Instructions for CNN model training

1. Clone the repository
```	
git clone https://github.com/udacity/dog-project.git
```

2. Install the following python libraries
```	
pip install keras
pip install tensorflow
pip install opencv-python
```

3. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  
Unzip the folder and place it in the repo, at location `path/to/project/dog_images`. 

4. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). 
Unzip the folder and place it in the repo, at location `path/to/project/data/lfw`.

5. Download the 
[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz), 
[Resnet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) and
[Xception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz).
Place it in the repo, at location `path/to/project/bottleneck_features`.

6. Run jupyter notebook, and open `dog_app.ipynb`
```	
jupyter notebook
```

7. Html file is exported as `dog_app.html`

### Project Instructions for web application

1. Run the following command in the "app" directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/

![Sample Output][image2]


## Refinement

The training was started with a fully self-defined model, which includes three Convolutional Layer, three Pooling Layer, 
a Dropout Layer and a Dense Layer. And it achieved an accuracy of 7.0574%. The guessing accuracy of finding 1 breed out of 
133 categories is < 1%. Therefore, the self-defined model is far better than guessing.

In the second stage, I implemented the model on top of four different pre-trained model (VGG19, Resnet50, InceptionV3, or Xception).
The best accuracy increases to 85.4067% when I worked with Xception pre-trained model.

Furthermore, trying different number of epoch by looking at the trend of accuracy and loss, helps in increasing the accuracy.

## Model Evaluation and Validation

Validation dog images are used in the training process to validate model accuracy after each epoch, and also it is useful 
in preventing over-fitting.

Testing dog images are set aside in the training process, and only use to validate training accuracy when the training is finished.

## Justification

Overall, the accuracy of detecting whether or not an image including a dog is 100% and 
the accuracy of telling a correct dog breed is 85%. 

It is difficult to tell why one model works better than the others, and there is no golden rule to tell how to set the number of epoch, 
learning rate and so on. In most cases, we can just try to train with different pre-trained model with different parameters to get 
the best results.

## Reflection

Having an accuracy of over 85% is better than I expected. One interesting thing while working on this project is the pre-trained 
model can significantly reduce the training computing work, and increase the accuracy, which is amazing.

## Future Improvement

The training dataset only has about 8,000 images, and we need to classify them over 100 categories.
 It means each category having less than 100 images, which is not enough. Having a more training data will definitely 
 improve the accuracy
 
 

