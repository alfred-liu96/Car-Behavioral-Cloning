# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around both track one and two without leaving the road

[//]: # (Image References)

[image1]: ./output/
[image2]: ./output/image1.jpg
[image3]: ./output/image2.jpg
[image4]: ./output/image3.jpg
[gif1]: ./output/run1.gif
[gif2]: ./output/run2.gif

---

### Project Structure

#### 1. Files Introduction

My project includes the following main files:

* behavioral_cloning_inception_v3.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network in Keras
* run1.mp4 a video recording of my vehicle driving autonomously around the track one
* run2.mp4 a video recording of my vehicle driving autonomously around the track two

#### 2. Simulator Driving
Using the [Udacity provided simulator](https://github.com/udacity/self-driving-car-sim "simulator") and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy of mine for deriving a model architecture was to use a state-of-the-art CNN model for transfer learning and fine-tuning the model to increase its generalization ability. So according to this strategy I chose Inception-V3 as my first choice and started a two-day long nightmare.

The Inception-V3 is a compelling model with few parameters. At first, to make the full use of it, I simply replaced the top fully-connected layer and the output layer, loaded the pre-trained weights of other layers and froze them, then I trained the new model with the data collected from the simulator. The training loss and validation loss were close but both high, and it indicated the new model was underfitting.

At this point, since the new added top layers were well trained and the model was underfitting, I could start fine-tuning several convolutional layers from Inception-V3.

I unfroze the parameters of the top 2 inception blocks to train with a low learning rate. This time the training loss was much smaller than the validation, and my model faced overfitting. 

To solve this problem I tried to add a dropout layer before the fully-connected layer, use data augmentations like image cropping and image flipping, change the architecture of the fully-connected layer or decrease the number of unfrozen layers from Inception-V3 to fine-tune, but none of them really worked.

After two days of trying, I finally realized that the problem is that I used an overly complicated model. So I decided to use a smaller Inception-V3 with only four inception blocks instead of an entire model (behavioral_cloning_inception_v3.py lines 104-133), and fine-tune the model over all of the layers instead of several of them (behavioral_cloning_inception_v3.py lines 157-175).

This time, the model performed better than ever (although I still faced the overfitting issue), and the vehicle was able to drive autonomously around both the track one and track two without leaving the road.

#### 2. Final Model Architecture

The final model architecture consists of a smaller Inception-V3 with only four bottom inception blocks followed by a global average pool, one dropout layer, two dense layers, and the output layer which predicts the steering angles.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on both track one and track two. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center on track one so that the vehicle would learn to drive to the center from roadsides.

I also drive in the opposite direction on track two to make the model generalize better.

For data agumentation, I flipped images and angles, below is an image that has then been flipped:

![alt text][image3]

Looking at the collected images, we can see that the lane lines always appear in a specific area of them, so we can crop each image to focus on only the portion of the image that is useful for predicting a steering angle.
Here is an example:

![alt text][image4]

#### 4. Video Test

![alt text][gif1]

![alt text][gif2]

#### 5. Issues

1. My car seems to like to drive on the edge of the lane line, and this may be caused by the feeding data. Since the track 2 is so tricky that I can't drive the car between the lane lines.
2. My car will run out of the lane when the driving speed is too high (over 30km/h) in track two.
