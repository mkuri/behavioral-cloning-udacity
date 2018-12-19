# **Behavioral Cloning** 

## Writeup Template

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[loss100]: ./outputs/loss_and_epoch_100_300px.png "Loss100"
[loss100drop]: ./outputs/loss_and_epoch_100_dropout_300px.png "Loss100dropout"
[loss1000]: ./outputs/loss_and_epoch_1000_300px.png "Loss1000"
[loss1000drop]: ./outputs/loss_and_epoch_1000_dropout_300px.png "Loss1000dropout"
[center]: ./outputs/center.jpg "center"
[left]: ./outputs/left.jpg "left"
[right]: ./outputs/right.jpg "right"
[bflip]: ./outputs/before_flip.jpg "before flip"
[aflip]: ./outputs/after_flip.jpg "after flip"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I built a learning model based on Nvidia's model. 

As preprocessing of the image, I normalized the pixel values to be [-0.5, 0.5]. Then, I cropped the images to remove the sky and the hood of the car.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 79-98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 124). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia's model (https://arxiv.org/abs/1604.07316). I thought this model might be appropriate because the model solved the similar problem of estimating the steering angle from images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add dropout layers between fully-connected layers.

![loss100][loss100] ![loss100drop][loss100drop]
![loss1000][loss1000] ![loss1000drop][loss1000drop]

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 79-98) is shown below.

|Layer|Property|Output|
|:----|:----|:----|
|Input|-|(90, 320, 3)|
|Conv 2D|kernel=(5, 5), strides=(2, 2), activation='relu'|(43, 158, 24)|
|Conv 2D|kernel=(5, 5), strides=(2, 2), activation='relu'|(20, 77, 36)|
|Conv 2D|kernel=(5, 5), strides=(2, 2), activation='relu'|(8, 37, 48)|
|Conv 2D|kernel=(3, 3), activation='relu'|(6, 35, 64)|
|Conv 2D|kernel=(3, 3), activation='relu'|(4, 33, 64)|
|Flatten|-|(8448)|
|Fully Connected|activation='relu'|(1000)|
|Dropout|probability=0.5|(100)|
|Fully Connected|activation='relu'|(100)|
|Dropout|probability=0.5|(100)|
|Fully Connected|activation='relu'|(50)|
|Dropout|probability=0.5|(100)|
|Fully Connected|activation='relu'|(10)|
|Fully Connected|-|(1)|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][center]

I then included images of the cameras on the left side and the right side of the vehicle in the training set to learn the steering angle which returns the vehicle to the middle from the left side or the right side.
When using the left camera image, I added 0.2 to the recorded steering angle. When using the camera image on the right side, I subtracted 0.2 from the recorded steering angle.

![left][left] ![right][right]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would be a value multiplied by -1. For example, here is an image that has then been flipped:

![before flip][bflip]
![after flip][aflip]

After the collection process, I had 8052 of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6- as evidenced by the following graph. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![loss1000drop][loss1000drop]
