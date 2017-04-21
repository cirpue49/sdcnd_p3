# **Transfer Learning for Behavioral Cloning** 

### I wrote a blog post about this project, please check out [this medium post](https://medium.com/@kosukemurakami/transfer-learning-for-behavioral-cloning-df0d49c0c69b).

---
[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/top_model.png "Grayscaling"
[image3]: ./examples/hist.png "Recovery Image"
[image4]: ./examples/center_2016_12_01_13_38_26_805.png "Recovery Image"
[image5]: ./examples/center_2016_12_01_13_33_44_096.png "Recovery Image"
[image6]: ./examples/preprocessed_hist.png "Normal Image"
[image7]: ./examples/left_img.png "Flipped Image"
[image8]: ./examples/center_img.png "Flipped Image"
[image9]: ./examples/right_img.png "Flipped Image"



#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md

#### 2. Submission includes functional code
Using the Udacity provided simulator with 'Good' graphics quality and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### 3. Submission code is usable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of pretrained vgg16 net and original fully connected layers. (model.py lines 263-284) 

The model includes ELU layers to introduce nonlinearity (code line 272, 277, 282), and the data is normalized when the model loads the images. (code line 50). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 274, 278, 283). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 138-164). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model based on transfer learning used a SGD optimizer, so the model would converge smoothly. (model.py line ).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Before augmenting provided data, I deleted the data with unappropriate behavior. In the process of augmenting data, I used multiple camera images, and change images' brightness randomly.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use transfer learning. All I had to think for the architecture was fully connected layers.

In my first step, I imitated the similar fully connected layer to VGG16 since my transfer learning is based on VGG16's pretrained model. But since it did not converge as I intended, I added batch normalization layer and reduced the number of paramators in fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the data so that the noise in the data would be discarded. For details about how I created the training data, see the next section.  


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially around the bridge. I noticed that black color of bridge could be ignored through the feature extraction step. So, I decided to fine-tuning the model to improve the driving behavior. By fine-tuning I mean, making a few of VGG16's weights trainable.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 263-284) consisted of a convolution neural network with VGG16 model and fully connected layers.

Here is a visualization of the architecture.

Note that the left picture represents VGG16 model and the right picture describes fully connected layers.

![alt text][image1]
![alt text][image2]

#### 3. Creation of the Training Set

I used the provided udacity dataset. 

I have noticed several problems in the data.

##### Problem 1: Distribution
The distribution of data is really biased to 0.

I need to prepare balanced data to mimic driving behavior otherwise the model would be biased to predict 0 angle.

![alt text][image3]

##### Problem 2: Unusual angle

The few picture has noise in the data. For example, this picture's angle is -0.923. If my model mimic this behavior, the car would be crashed. 

Stamp: center_2016_12_01_13_38_26_805.jpg

######   Angle: -0.923 
![alt text][image4]

##### Problem 3: Spiky data

The curves in provided data tend to have spikes in its distribution. In order to achieve smooth turn, I needed to delete such spikes.

![alt text][image5]

[Sample Video with Steering](https://www.youtube.com/watch?v=EVK0-hhxx8Y&feature=youtu.be)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/EVK0-hhxx8Y/0.jpg)](https://www.youtube.com/watch?v=EVK0-hhxx8Y)

#### Solutions to the problems

1: Cleaning data

I cropped steering angles from -0.4 to 0.4 to remove the spikes in the data.

```sh
df = pd.read_csv(csv_path,index_col = False)
df = df[df.steering > -0.4][df.steering < 0.4]
```

In addtion, I manually deleted some spiky parts.

Note that red frames are deleted and green ones are kept. 

[Preprocessed Video](https://www.youtube.com/watch?v=H1-hO4ZzH4Q&feature=youtu.be)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/H1-hO4ZzH4Q/0.jpg)](https://www.youtube.com/watch?v=H1-hO4ZzH4Q)

2: Balanced data

Since the original data is biased toward 0, it needs to be somehow balanced. 

I divided the data into 8 bins(-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4) and made 1 batch with 160 samples, together with 20 samples for each bin. 

This is the histogram for 1 batch, 160 samples. Notice that this histogram has the values above 0.4 and below -0.4. This is due to the data augmentation I implemented. The details will be explained in next session. 

![alt text][image6]

#### Data Augmentation

1: Put gaussian noise on steering noise 

Add a little bit gaussian noise on steering angles with mean 0, std 0.01.

2: Using multiple cameras 

Used left and right camera images to keep the car located in center. 

Left, Center, Right Camera images with 0 steering angle. 

![alt text][image7]
![alt text][image8]
![alt text][image9]

The number of pixcel to offset from an original steering angle is hyper paramator. 

I offset 0.05 for right and left camera images since the above pictures look really smilar to each other. 

```sh
if left_image:
   shift_angle = .05 + gaussian_noise(mean = 0.0, std = 0.01)
elif center_image:
   shift_angle = 0. + gaussian_noise(mean = 0.0, std = 0.01)
elif right_image:
   shift_angle = -.05 - gaussian_noise(mean = 0.0, std = 0.01)
steering = original_steering + shift_angle
```

[Result Video Track1](https://www.youtube.com/watch?v=rg2aTWEvBz4)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/rg2aTWEvBz4/0.jpg)](https://www.youtube.com/watch?v=rg2aTWEvBz4)


