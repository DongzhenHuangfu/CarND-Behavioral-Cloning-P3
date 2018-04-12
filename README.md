# **Behavioral Cloning** 

## Instruction

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 3 and 64 (model.py lines 61-64) 

The model includes RELU layers to introduce nonlinearity (code line 61-64), and the data is normalized in the model using a Keras lambda layer (code line 58). 

#### 2. Reduce overfitting in the model

The model was trained and validated on different data sets(0.3 of the data serve as the validation set) to ensure that the model was not overfitting (code line 22-55). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road so that the cars will know when to turn hardly avoiding driving out of the track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to combine the convolutional layer with fully connected layer so that the model is more flexible for the training.

My first step was to use a convolution neural network model similar to the architecture which NVIDIA used for training the driverless car. I thought this model might be appropriate because it has already been widely used and proved to be effective.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had relevant similar low mean squared error on the training set and the validation set. This implied that the model was great. Here is the result of my training process, it represents the variation of the loss with epochs:

<img width="50%" height="50%" src="https://github.com/DongzhenHuangfu/CarND-Behavioral-Cloning-P3/raw/master/pictures/loss.jpg"/>

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and the car was to near to the right side of the track.To improve the driving behavior in these cases, I added some data in which I control the car starting on the right side and tried to go to the middle of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 57-69) consisted of a convolution neural network with the following fully connected layers.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

<img width="40%" height="40%" src="https://github.com/DongzhenHuangfu/CarND-Behavioral-Cloning-P3/raw/master/pictures/center.jpg"/>

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center of the road from left or the right side. These images show what a recovery looks like starting from left or the right side of the road :

<img width="40%" height="40%" src="https://github.com/DongzhenHuangfu/CarND-Behavioral-Cloning-P3/raw/master/pictures/right1.jpg"/>        
<img width="40%" height="40%" src="https://github.com/DongzhenHuangfu/CarND-Behavioral-Cloning-P3/raw/master/pictures/left.jpg"/>        
<img width="40%" height="40%" src="https://github.com/DongzhenHuangfu/CarND-Behavioral-Cloning-P3/raw/master/pictures/right2.jpg"/>

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would be helpful to elimate the influence of the track shape, cause in this software, the track is always turning to the same side. For example, here is an image that has then been flipped:

<img width="40%" height="40%" src="https://github.com/DongzhenHuangfu/CarND-Behavioral-Cloning-P3/raw/master/pictures/right2.jpg"/>         
<img width="40%" height="40%" src="https://github.com/DongzhenHuangfu/CarND-Behavioral-Cloning-P3/raw/master/pictures/right2_flip.jpg"/>

After the collection process, I had 15706 number of data points. I then preprocessed this data by cutting out the unnecessary part of the picture to save the memory(see line 60 in model.py).


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by the result I get when I setted the epochs as 20, I found that the loss stop to reduce at about epoch 7 or 8. I used an adam optimizer so that manually training the learning rate wasn't necessary.
