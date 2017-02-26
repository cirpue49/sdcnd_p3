#Import 
import os
import cv2
import math
import numpy as np
import pandas as pd
from numpy.random import *
#Import the keras layers
import keras
from keras import initializations
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.models import model_from_json, load_model, Model, Sequential
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Dense
from keras.utils.visualize_util import plot

"""
CONSTANTS
"""
PATH = 'data/'  # Data path

# Data augmentation constants
OFF_CENTER_IMG = .05  # Angle change when using off center images
BRIGHTNESS_RANGE = .25  # The range of brightness changes
GAUSS_NOISE = 0.01 #Standard deviation for gaussian distribution

# Training constants
BATCH = 80  # Number of images per batch
TRAIN_BATCH_PER_EPOCH = 200  # Number of batches per epoch for training
EPOCHS = 10  # Minimum number of epochs to train the model on

# Image constants
IMG_ROWS = 64  # Number of rows in the image
IMG_COLS = 64  # Number of cols in the image
IMG_CH = 3  # Number of channels in the image

"""
FUNCTIONS
"""
def img_pre_process(image):
    #Shape = (160, 320)
    shape = image.shape
    #Crop an image, height 40:125, width 0:320
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(IMG_ROWS,IMG_COLS), interpolation=cv2.INTER_AREA)
    #Normalise an image -0.5 to 0.5
    image = image/255. - .5
    return np.resize(image, (1, IMG_ROWS, IMG_COLS, IMG_CH))


def img_change_brightness(img):
    temp = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = BRIGHTNESS_RANGE + np.random.uniform()
    temp[:,:,2] = temp[:,:,2]*random_bright
    image = cv2.cvtColor(temp,cv2.COLOR_HSV2RGB)
    return image

def pre_process_bin(bin_data):
    r_index = np.random.randint(0,len(bin_data))
    c_ref = np.random.randint(3)
    
    #Using multiple camera images
    #Depending on a camera, shift angle and put noise 
    if (c_ref == 0):
        path_file = os.path.join(PATH, bin_data.reset_index()['left'][r_index].strip())
        shift_ang = OFF_CENTER_IMG + normal(0., GAUSS_NOISE)
    if (c_ref == 1):
        path_file = os.path.join(PATH, bin_data.reset_index()['center'][r_index].strip())
        shift_ang = normal(0, GAUSS_NOISE)
    if (c_ref == 2):
        path_file = os.path.join(PATH, bin_data.reset_index()['right'][r_index].strip())
        shift_ang = -OFF_CENTER_IMG - normal(0., GAUSS_NOISE)
    y_steer = bin_data.reset_index()['steering'][r_index] + shift_ang
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = img_change_brightness(image)
    
    #Flip image 
    f_ref = np.random.randint(2)
    if (f_ref == 0):
        image = cv2.flip(image, 1)
        y_steer = -y_steer
        
    image = img_pre_process(image)
    image = np.array(image)
    return image,y_steer



def val_data_generator(batch_df):
    assert len(batch_df) == BATCH, 'The length of the validation set should be batch size'
    while 1:
        _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
        _y = np.zeros(BATCH, dtype=np.float)

        for idx in np.arange(BATCH):
            _x[idx] = img_pre_process(cv2.imread(os.path.join(PATH, batch_df.center.iloc[idx].strip())))
            _y[idx] = batch_df.steering.iloc[idx]
        yield _x, _y

def train_data_generator(df):
    _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
    _y = np.zeros(BATCH, dtype=np.float)
    while 1:
        #BATCH = 80
        #1 batch is composed of 8 bins
        #10 images for each bin
        x_bins = np.zeros((8, 10, IMG_ROWS, IMG_COLS, IMG_CH))
        y_bins = np.zeros((8, 10))
        for bindex in range(8):
            lower = df.loc[df.steering >= -1 + ((bindex+6)/10)]
            bin_df = lower.loc[lower.steering < -1 + ((bindex+7)/10)]
            bin_images = np.zeros((10, IMG_ROWS, IMG_COLS, IMG_CH))
            bin_steerings = np.zeros(10)
            for i in range(10):
                assert len(bin_df) != 0, 'bin_df has nothing in here'
                image, steering = pre_process_bin(bin_df)
                bin_images[i] = image
                bin_steerings[i] = steering
            x_bins[bindex] = bin_images
            y_bins[bindex] = bin_steerings
                
            if bindex == 0:
                _y = y_bins[bindex]
                _x= x_bins[bindex]
            else:
                _y = np.concatenate((_y,y_bins[bindex]), axis=0)
                _x = np.concatenate((_x,x_bins[bindex]), axis=0)
        yield _x, _y
        # Reset the values back
        _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
        _y = np.zeros(BATCH, dtype=np.float)
        

def train_model(model, train_data, val_data):
    # Get an evaluation on the validation set
    val_loss = model.evaluate_generator(val_data_generator(val_data), val_samples=BATCH)
    print('Pre-trained evaluation loss = {}'.format(val_loss))

    # Try some predictions before we start..
    test_predictions(model, train_data)

    num_runs = 0
    while True:

        print('Run {}'.format(num_runs+1))

        history = model.fit_generator(
            train_data_generator(train_data),
            samples_per_epoch=TRAIN_BATCH_PER_EPOCH * BATCH,
            nb_epoch=1,
            validation_data=val_data_generator(val_data),
            nb_val_samples=BATCH*5,
            verbose=1)
        num_runs += 1

        # Print out the test predictions
        test_predictions(model, train_data)
        save_model(model, num_runs)
        if num_runs > EPOCHS-1:
            break


def test_predictions(model, df, num_tries=8):
    print('Predictions: ')
    for bindex in np.arange(num_tries):
        lower = df.loc[df.steering > -1 + ((bindex+6)/10)]
        bin_df = lower.loc[lower.steering < -1 + ((bindex+7)/10)]
        rnd_idx = np.random.randint(0,len(bin_df))
        img = img_pre_process(cv2.imread(os.path.join(PATH, bin_df.center.iloc[rnd_idx].strip())))
        img = np.resize(img, (1, IMG_ROWS, IMG_COLS, IMG_CH))
        org_angle = bin_df.steering.iloc[rnd_idx]
        pred_angle = model.predict(img, batch_size=1)
        print(org_angle, pred_angle[0][0])


def save_model(model, epoch=''):
    model.save_weights('model'+str(epoch)+'.h5')
    model.save('model'+str(epoch))
    print('Model saved')
    
def prepare_df(df):
    #Cut useless images
    temp1 = df[63:87]
    temp2 = df[98:126]
    temp3 = df[167:1172] 
    temp4 = df[1231:1276]
    temp5 = df[1280:1635]
    temp6 = df[1665:1724]
    temp7 = df[1754:2375]
    temp8 = df[2395:3002]
    temp9 = df[3007:3378]
    temp10 = df[3382:3431]
    temp11 = df[3461:3541]
    temp12 = df[3560:3760]
    temp13 = df[3764:3896]
    temp14 = df[3925:4069]
    temp15 = df[4093:4113]
    temp16 = df[4142:4277]
    temp17 = df[4282:4318]
    temp18 = df[4320:4352]
    temp19 = df[4357:4533]
    temp20 = df[4648:5288]
    temp21 = df[5307:5328]
    temp22 = df[5337:5467]
    temp23 = df[5472:5842]
    temp24 = df[5879:6058]
    temp25 = df[6097:6287]
    temp26 = df[6293:6718]
    temp27 = df[6737:7146]
    temp28 = df[7170:7221]
    temp29 = df[7279:7319]
    temp30 = df[7348:7359]
    temp31 = df[7378:7398]
    temp32 = df[7427:7438]
    temp33 = df[7486:7507]
    temp34 = df[7535:7555]
    temp35 = df[7605:]
    
    #Making noise-free data frame 
    nf_df = temp1.append(temp2)
    nf_df = nf_df.append(temp3)
    nf_df = nf_df.append(temp4)
    nf_df = nf_df.append(temp5)
    nf_df = nf_df.append(temp6)
    nf_df = nf_df.append(temp7)
    nf_df = nf_df.append(temp8)
    nf_df = nf_df.append(temp9)
    nf_df = nf_df.append(temp10)
    nf_df = nf_df.append(temp11)
    nf_df = nf_df.append(temp12)
    nf_df = nf_df.append(temp13)
    nf_df = nf_df.append(temp14)
    nf_df = nf_df.append(temp15)
    nf_df = nf_df.append(temp16)
    nf_df = nf_df.append(temp17)
    nf_df = nf_df.append(temp18)
    nf_df = nf_df.append(temp19)
    nf_df = nf_df.append(temp20)
    nf_df = nf_df.append(temp21)
    nf_df = nf_df.append(temp22)
    nf_df = nf_df.append(temp23)
    nf_df = nf_df.append(temp24)
    nf_df = nf_df.append(temp25)
    nf_df = nf_df.append(temp26)
    nf_df = nf_df.append(temp27)
    nf_df = nf_df.append(temp28)
    nf_df = nf_df.append(temp29)
    nf_df = nf_df.append(temp30)
    nf_df = nf_df.append(temp31)
    nf_df = nf_df.append(temp32)
    nf_df = nf_df.append(temp33)
    nf_df = nf_df.append(temp34)
    nf_df = nf_df.append(temp34)
    
    return nf_df[df.steering > -0.4][df.steering < 0.4][df.throttle > .25]

def get_model():
    #prepare vgg16 model
    input_tensor = Input(shape=(64, 64, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    #FC layers
    act = keras.layers.advanced_activations.ELU(alpha=1.0)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(512, init='he_normal'))
    top_model.add(BatchNormalization())
    top_model.add(act)
    act = keras.layers.advanced_activations.ELU(alpha=1.0)
    top_model.add(Dropout(0.5))
    top_model.add(Dense(128, init='he_normal'))
    top_model.add(BatchNormalization())
    top_model.add(act)
    top_model.add(Dropout(0.5))
    top_model.add(Dense(32, init='he_normal'))
    top_model.add(BatchNormalization())
    act = keras.layers.advanced_activations.ELU(alpha=1.0)
    top_model.add(act)
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, name = 'output', init='he_normal'))
    
    #building a model based on vgg16 and original FC layers
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
    
    #print trainable and untrainable paramators
    for layer in model.layers[:15]:
        print ("untrainable", layer)
        layer.trainable = False
    for layer in model.layers[16:19]:
        print ("trainable", layer)
    for layer in top_model.layers:
        print("trainable", layer)
        
    model.compile(optimizer=SGD(lr=1e-4,  momentum=0.9), loss='mse')
    model.summary()
    
    #Make visualisation for the model
#     plot(model, to_file='model.png')
#     plot(top_model, to_file='top_model.png')
    return model

if __name__ == '__main__':
    # Set the seed for predictability
    np.random.seed(200)

    # Load the data
    original_df = pd.read_csv(os.path.join(PATH, 'driving_log.csv'))

    # Crop the data
    cropped_df = prepare_df(original_df)
    
    # Shuffle and split the data set
    validate, train = np.split(cropped_df.sample(frac=1), [BATCH])
    del total_data

    # Create a model
    steering_model = get_model()

    # Train the model
    train_model(steering_model, train, validate)

    exit(0)

