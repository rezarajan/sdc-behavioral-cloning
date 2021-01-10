# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_files/data_distribution.png "Data Distribution"
[image2]: ./report_files/training_loss.png "Training Loss"
[image3]: ./report_files/fine_tuning.png "Fine-Tuning Training Loss"
[image4]: ./report_files/dirt_maneuver.jpg "Differentiating Roads from Dirt"
[image5]: ./report_files/edge_maneuver.jpg "Maneuvering from Edges"
[image6]: ./report_files/post_maneuver.jpg "Steering Away from Posts"
[gif1]: ./report_files/return_to_lane.gif "Returning to Lane"
[gif2]: ./report_files/bridge.gif "Lane Keeping on Bridge"
[gif3]: ./report_files/dirt_road.gif "Distinguishing Road from Dirt"
[gif4]: ./report_files/lane_keeping.gif "Lane Keeping"
[gif5]: ./report_files/hairpin_offset.gif "Re-adjusting for a Hairpin Turn"
[gif6]: ./report_files/hairpin_hard.gif "Re-adjusting for a Very Tight Hairpin Turn"
[gif7]: ./report_files/curved_lane_keeping.gif "Lane Keeping on a Curved Road"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioral_cloning.ipynb containing the code to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* report.md summarizing the results
* output_video.mp4 for both track 1 and track 2, showcasing autonomous runs. These videos can be found under the *recordings/track\*/* folder, for each respective track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The behavioral_cloning.ipynb Jupyter notebook contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The general structure of the model used in this project is similar to the NVIDIA's [model](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), but uses:

* Max-Pooling after each convolution layer
* Smaller fully-connected layers
* Two dropout layers and
* Only 1 output for steering angle
_________________________________________________________________
|Layer (type)                 |Output Shape         |     Param #|   
|-----------------------------|---------------------|------------|
|lambda (Lambda)              |(None, 128, 256, 1)  |     0      |  
|cropping2d (Cropping2D)      |(None, 78, 256, 1)   |     0      |  
|conv2d (Conv2D)              |(None, 37, 126, 24)  |     624    |  
|max_pooling2d (MaxPooling2D) |(None, 36, 125, 24)  |     0      |  
|conv2d_1 (Conv2D)            |(None, 16, 61, 36)   |     21636  |  
|max_pooling2d_1 (MaxPooling2 |(None, 15, 60, 36)   |     0      |  
|conv2d_2 (Conv2D)            |(None, 6, 28, 48)    |     43248  |  
|conv2d_3 (Conv2D)            |(None, 4, 26, 64)    |     27712  |  
|max_pooling2d_2 (MaxPooling2 |(None, 3, 25, 64)    |     0      |  
|conv2d_4 (Conv2D)            |(None, 1, 23, 64)    |     36928  |  
|flatten (Flatten)            |(None, 1472)         |     0      |  
|dropout (Dropout)            |(None, 1472)         |     0      |  
|dense (Dense)                |(None, 600)          |     883800 |  
|dense_1 (Dense)              |(None, 100)          |     60100  |  
|dense_2 (Dense)              |(None, 50)           |     5050   |  
|dropout_1 (Dropout)          |(None, 50)           |     0      |  
|dense_3 (Dense)              |(None, 1)            |     51     |  
_________________________________________________________________

* Total params: 1,079,149
* Trainable params: 1,079,149
* Non-trainable params: 0

The methodology of this model follows a funnel-type structure, where information is gradually filtered. Therefore, the model starts with larger convolutions with filter size 5, to filter for larger features of each frame. Subsequently, smaller convolution filters of size 3 are used to capture more granular, but important details of the frame.

The model also includes RELU activations to introduce nonlinearity, and dropouts to reduce overfitting. Furthermore, data is normalized in the first layer of the model, using a Lambda layer.

Details of the model architecture may be found in the "Model Architecture" section of the Jupyter notebook. 

---
#### 1.5 Image Preprocessing
Aside from image normalization in the Lambda layer, all training data is pre-processed using the following methods:

* Contrast Limited Adaptive Histogram Equalization (CLAHE): reduces the effect of lighting changes in the model
* Grayscaling: from testing, color images do not necessarily improve model performance, and grayscaling helps to reduce model size
* Image scaling: to further reduce model size (and the number of parameters to train) the image is downscaled, while maintaining its aspect ratio

These filters are also added to the drive.py file to process each frame before model prediction.

---
#### 2. Attempts to reduce overfitting in the model

As previously mentioned, the model uses two dropout layers to reduce overfitting. Specifically, the first dropout layer has a dropout rate of 0.25, and the second 0.50, following the filter-type structure.

Furthermore, the data is suffled on each epoch, and separated into training and validation sets to ensure that it is not fitting to any particular subset of the data.

Ultimately, the evaluation of model generalization can be tested by running it on different tracks, i.e. in different environments. Therefore, the model is also trained and tested on data from track 2.

Model overfitting during training is also observed using a stopper, where the loss metric is evaluated after each epoch, to ensure that a minimum difference between the current and previous loss values are met. If not, then after a set number of attempts (called patience), the training is stopped. This ensures that the loss is always decreasing, and training stops when it does not, to avoid overfitting.

---
#### 3. Model parameter tuning

Parameters such as layer shapes, convolution kernel sizes and dropout rates are tuned based on:

1. Computational capability of the GPU
2. Model loss after training and
3. Simlated tests of the model on the track

The optimizer used is the adam optimizer, and therefore learning rate is not specified.

---
#### 4. Appropriate training data

Training data is chosen with the following principles in mind:

* Driving strictly with the vehicle on the road
* Maneuvering from situations where the vehicle may deviate from the road, or moves out of lane
* Balancing the distribution of left and right turns (negative and positive steering angles) such that the data is unbiased
* Exposure to different driving terrain and environments

Below is a visualization of the distribution of data based on steering angles:

![image1]
 
 The distribution appears to be centered around the 0.0 steering angle, with positive and negative steering angles similarly distributed. The inequitable distribution of smaller steering angles to the extremes may introduce bias in the trained model, but to preserve data these are not removed.

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The primary basis of model design follows from the success of the previously mentioned NVIDIA [model](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). However, due to the large model size, changes are made to reduce training time on the available hardware, based on the following ideas:

* Use convolution layers to extract relevant features from each frame (assumed to be the drivable area, i.e. the road)
* Use sufficiently large filter sizes and strides such that relevant data is retained, but subsequent layer sizes are reduced
* Use max pooling to retain relevant data while reducing layer sizes 
* Keep the model structure deep, rather than wide, i.e. hold a preference to adding more, but smaller layers rather than increasing layer size
* Introduce non-linearities through RELU activations
* Introduce dropout layers to reduce overfitting

---

**Generators:**

During training, generators are used to reduce the memory overhead required to load the training and validation datasets. In fact, without using generators to load and process images on-the-fly, the model cannot be trained due to memory overflow.

---

**Training:**

Training was performed using the following parameters:

* Batch Size: 512
* Epochs: 30
* Loss Metric: Mean Squared Error (Training Set)
* Stopper: 
  * Mininim loss delta: 0.0003
  * Patience: 5
* Validation Split: 0.20

Note that the loss metric is on the training set, and not the validation set. This is because in practice, the model actually underfits the training set, i.e. the training set has a consistently higher loss than the validation set. Initial attempts used the validation set as a loss metric, but the resulting model did not produce as desirable of results as is hoped. Furthermore, it is observed that using the training set loss did not negatively affect the validation set loss.

Below is a visualization of the model's loss after each epoch:

![image2]

Training is stopped after 25 epochs, by the stopper, which is approximately at the point the training and validation sets have similar losses. This is suitable as it prevents overfitting the data, while allowing the model to train enough such that it also does not underfit.

---

**Fine-Tuning:**

After training the model is tested on tracks 1 and 2. For the most part, it completes track 1 at this point, but it does make some unexpected deviations at certain corners. Furthermore on track 2, it fails to complete major hairpin turns, as the car understeers. When it goes off-track it also struggles to maneuver back in certain points of track 2. Therefore, it is clear that the model needs to be trained on more data pertaining to turning tight corners, as well as in cases where it needs to maneuver back on track.

However, due to time constraints, it is not feasible to re-train the model from scratch. Therefore, the previously trained model is trained on more data pertaining to the previously mentioned scenarios, assuming the initial weights are closer than if it were to be re-initialized.

In a similar manner, the training resulted in the following loss graph:

![image3]

The above graph also indicates model underfitting. In this case, however, the validation set's loss remained fairly constant, while the training set loss reduces.

This model is now able to complete both tracks 1 and 2 without deviating from the road, while, for the most part, staying in lane in track 2. Furthermore, on track 1 it tends to stick to the right part of road, though it is unmarked.

---
#### 2. Final Model Architecture

The final model architecture is the same as what had been started with as seen [previously](#model-architecture-and-training-strategy). At the end of training the most notable difference that resulted in success is the dataset used, which goes to show how robust convolutional neural nets can be.

---
#### 3. Creation of the Training Set & Training Process

As noted previously, the performance of the model depends more on the quality of data rather than on the intricacies model. Therefore, capturing high-quality data involves the following:

* Consistent driving behavior
* Capturing a variety of scenarios and environments
* Capturing maneuvers during edge cases
* Creating a well-distributed data set (i.e. equitably distributed)

Twenty-three (23) recording sessions are captured to properly address each of the above points. However, in some recordings it may be said that the driving patterns are not consistent, and this has to do with the method of driving, i.e. using a mouse rather than a joystick.

As a baseline the car is driven around the tracks once. Then, since track 1 has a notable left-turn bias, it is driven around the track in the opposite direction. The same is done for track 2, but in this case it is driven in reverse to account for the downhill bias. 

In all sessions, three images are captured: a left, center and right camera image. This is done to increase the quantity of data captured, as is required for training neural networks. The steering angles are adjusted for the left and right images to account for the offset.

Augmentations are not performed on the dataset, though this may be an area for improvement. Image flips have been tested, with adjusted steering angles, but in testing this has proven to reduce model performance in simulation; driving the track in reverse seemed to yield better results.

Some examples of data acquisition are shown below:

![image4]

The above image shows the car driving near dirt, which in some cases depending on the angle of approach, the initial model mistakes as a road.

![image5]

The above image shows the car driving near an edge, where the intial model would sometimes drive off in sharp turns. Here, the initial model is drawn towards the posts, causing it to understeer.

![image6]

As an additional training scenario, the model is trained on a straight road to avoid the posts, which, as previously mentioned, it is drawn toward.

The above images show some scenarios in which the initial model struggles, and which have been used to fine-tune the model.

After data collection, the data is [processed](#15-image-preprocessing) and [fed into the model](#1-solution-design-approach) as mentioned previously.

---
### Results

Below are some scenarios in which the fine-tuned model is displaced from its original path. The scenarios show the ability of the model to correct itself and resume driving.

![gif1]

The above animation shows the car returning to the right side of the unmarked track when displaced.

![gif2]

The above animation shows the car re-aligning itself when displaced on the bridge.

![gif3]

The above animation shows how the model properly differentiates the road from the dirt, even when pushed closer to it.

![gif4]

The above animation shows how the model can properly distinguish between a left and right lane - and it finds the right lane when displaced.

![gif5]

The above animation shows that the model can find the right lane even around a sharp turn.

![gif6]

The above animation shows how the model can find the right lane around a tight hairpin turn - and it adjusts its steering accordingly to not run off the edge.

![gif7]

The above animation shows the model finding the right lane around an uphill turn.

---

### Recommendations

While the model completes tracks 1 and 2, and exhibits some level of understanding for driving by keeping to the right lane, there are still areas in which it makes unexpected decisions and veers into the oncoming lane. The following are recommended for further testing and improving the model:

* Gather driving data from different simulated environments
* Create a more equitable distribution of steering angles for the training data
* Improve steering angle offset for the left and right camera angles
* Use a joystick for smoother turns
* Utilize other image augmentation techniques to generate more data which the model may generalize upon

---
### Conclusions

A deep learning network has been successfully designed, trained and implemented in a simulation in which it drives a car across two different tracks, without deviating from the road. The robustness of the model to external deviations or unfamiliar scenarios is also showcased, and recommedations are provided for further testing and improvement.
