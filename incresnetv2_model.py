import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras import applications
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
import tensorflow as tf


# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'
# Set the numpy seed
np.random.seed(111)
# Disable multi-threading in tensorflow ops
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# Set the random seed in tensorflow at graph level
tf.set_random_seed(111)
# Define a tensorflow session with above session configs
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# Set the session in keras
K.set_session(sess)
# Make the augmentation sequence deterministic
aug.seed(111)

total_train_imgs = 5098
total_val_imgs = 519

def get_hyperparameters():
    pass

# Define hyperparameters
img_width, img_height = 299, 299
train_data_dir = Path("chest_xray/images/train")
val_data_dir = Path("chest_xray/images/val")
test_data_dir = Path("chest_xray/images/test")
batch_size = 16
nb_epochs = 10
nb_train_steps = total_train_imgs / nb_epochs
nb_val_steps = total_val_imgs


def get_image_data(data_dir):
    # Dirs
    norm_dir = data_dir / 'NORMAL'
    bact_dir = data_dir / 'BACTERIA'
    viral_dir = data_dir / 'VIRUS'
    # Get the list of all the images
    normal_cases = norm_dir.glob('*.jpeg')
    bacterial_cases = bact_dir.glob('*.jpeg')
    viral_cases = viral_dir.glob('*.jpeg')

    # Initialise lists to put all the images in, along with their labels: (img, label)
    dat = []
    # Add images to its list and label them: No-Pneumonia: 0, Bacterial: 1, Viral: 2
    for img in normal_cases:
        dat.append((img, 0))
    for img in bacterial_cases:
        dat.append((img, 1))
    for img in viral_cases:
        dat.append((img, 2))

    print("Got all image data from ", data_dir)
    return dat

# Get lists
print("Getting all image lists.")
train_dat = get_image_data(train_data_dir)
val_data  = get_image_data(val_data_dir)
test_data  = get_image_data(test_data_dir)


# Convert to pandas data frame
train_data = pd.DataFrame(train_dat, columns=['image', 'label'], index=None)
print("Completed getting all image lists.")

# Augmentation sequence 
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=20), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness

def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 299, 299, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 3), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Initialize a counter
    i =0
    while True:
        np.random.shuffle(indices)
        # Get the next batch 
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (299,299))
            
            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.
            
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            
            # generating more samples of the undersampled class
            if label==0 and count < batch_size-2:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.

                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                count +=2
            
            else:
                count+=1
            
            if count==batch_size-1:
                break
            
        i+=1
        print(batch_data)
        print(batch_labels)
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0


def decode_imgs_to_data(cases):
    # Initialise lists
    dat = []
    labels = []
    # Append all images to dat and labels
    counter = 0
    for img in cases:
        label = to_categorical(img[1], num_classes=3)
        img = mimg.imread(str(img[0]))
        img = cv2.resize(img, (299,299))
        if len(img) <= 2:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        dat.append(img)
        labels.append(label)
        counter += 1
        if (counter % 100 == 0):
            print(counter, "from class", label)

    # Convert the list into numpy arrays
    dat = np.array(dat)
    labels = np.array(labels)

    print("Decoded all images to data")
    return dat, labels

print("Decoding all imgs to data.")
# Get a train data generator
train_data_gen = data_gen(data=train_data[:100], batch_size=batch_size)

# Define the number of training steps
nb_train_steps = train_data.shape[0]//batch_size

val_data, val_labels = decode_imgs_to_data(val_data)
print("Finished decoding all imgs to data.")

# print(f"norm:\n {tr_norm[0][1]}")
# print(f"bact:\n {tr_bact[0][1]}")
# print(f"viral:\n {tr_viral[0][1]}")

# print(f"norm len: {len(tr_norm[0])}")
# print(f"bact len: {len(tr_bact[0])}")
# print(f"viral len: {len(tr_viral[0])}")

# print(f"norm:\n {tr_data[0]}")
# print(f"labels:\n {tr_labels[0]}")
# print(f"norm len: {len(tr_data[0])}")
# print(f"labels len: {len(tr_labels[0])}")



def get_image_generators():
    # Decompose images
    test_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    # Create set generators
    train_gen = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size)
    validation_gen = val_datagen.flow_from_directory(
            val_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size)
    test_gen = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size)

    return train_gen, validation_gen, test_gen

#train_generator, validation_generator, test_generator = get_image_generators()

def get_image_files(image_dir):
  fs = glob("{}/*.jpeg".format(image_dir))
  fs = [os.path.basename(filename) for filename in fs]
  return sorted(fs)

def create_model():
    print("Start creating model")
    # Get pretrained model
    model = applications.inception_resnet_v2.InceptionResNetV2(
        include_top=True,
        weights='imagenet',
        pooling='avg'
    )
    # Freeze layers
    for layer in model.layers:
        layer.trainable = False

    # Add trainable layers to the model
    x = model.output
    print("model shape")
    print(x.shape)
    #x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5, name='dropout2')(x)
    print("input to softmax shape")
    print(x.shape)
    predictions = Dense(3, activation='softmax')(x)

    # Create the final model and compile it
    final_model = Model(inputs=model.input, outputs = predictions)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    final_model.fit_generator(
        train_data_gen,
        steps_per_epoch = nb_train_steps,
        epochs = nb_epochs,
        validation_data = (val_data, val_labels),
        #callbacks = [checkpoint, early]
    )

    return final_model

res_model = create_model()
#res_model.summary()