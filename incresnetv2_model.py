
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
train_data_dir = Path('/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/pneumonia-babies/chest_xray/images/train')
val_data_dir = Path('/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/pneumonia-babies/chest_xray/images/val')
test_data_dir = Path('/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/pneumonia-babies/chest_xray/images/test')
batch_size = 16
nb_epochs = 10
nb_train_steps = total_train_imgs / nb_epochs
nb_val_steps = total_val_imgs

def get_train_image_data():
    # Dirs
    norm_dir = train_data_dir / 'NORMAL'
    bact_dir = train_data_dir / 'BACTERIA'
    viral_dir = train_data_dir / 'VIRUS'
    # Get the list of all the images
    normal_cases = norm_dir.glob('*.jpeg')
    bacterial_cases = bact_dir.glob('*.jpeg')
    viral_cases = viral_dir.glob('*.jpeg')

    # Initialise lists to put all the images in, along with their labels: (img, label)
    normal_data = []
    bact_data = []
    viral_data = []
    # Add images to its list and label them: No-Pneumonia: 0, Bacterial: 1, Viral: 2
    for img in normal_cases:
        normal_data.append((img, 0))
    for img in bacterial_cases:
        bact_data.append((img, 1))
    for img in viral_cases:
        viral_data.append((img, 2))


    normal_data_list = normal_data
    bact_data_list = bact_data
    viral_data_list = viral_data

    # # Convert to pandas data frame
    # normal_data = pd.DataFrame(normal_data, columns=['image', 'label'], index=None)
    # bact_data = pd.DataFrame(bact_data, columns=['image', 'label'], index=None)
    # viral_data = pd.DataFrame(viral_data, columns=['image', 'label'], index=None)
    # # Randomise the order
    # normal_data_list = normal_data.sample(frac=1.).reset_index(drop=True)
    # bact_data_list = bact_data.sample(frac=1.).reset_index(drop=True)
    # viral_data_list = viral_data.sample(frac=1.).reset_index(drop=True)
    # Check the data frame
    # print("train data")
    # print(f"normal:\n {normal_data_list.head()}")
    # print(f"bacterial:\n {bact_data_list.head()}")
    # print(f"viral:\n {viral_data_list.head()}")

    return normal_data_list, bact_data_list, viral_data_list

def get_val_image_data():
    # Dirs
    norm_dir = val_data_dir / 'NORMAL'
    bact_dir = val_data_dir / 'BACTERIA'
    viral_dir = val_data_dir / 'VIRUS'
    # Get the list of all the images
    normal_cases = norm_dir.glob('*.jpeg')
    bacterial_cases = bact_dir.glob('*.jpeg')
    viral_cases = viral_dir.glob('*.jpeg')

    # Initialise lists to put all the images in, along with their labels: (img, label)
    normal_data = []
    bact_data = []
    viral_data = []
    # Add images to its list and label them: No-Pneumonia: 0, Bacterial: 1, Viral: 2
    for img in normal_cases:
        normal_data.append((img, 0))
    for img in bacterial_cases:
        bact_data.append((img, 1))
    for img in viral_cases:
        viral_data.append((img, 2))


    normal_data_list = normal_data
    bact_data_list = bact_data
    viral_data_list = viral_data

    # # Convert to pandas data frame
    # normal_data = pd.DataFrame(normal_data, columns=['image', 'label'], index=None)
    # bact_data = pd.DataFrame(bact_data, columns=['image', 'label'], index=None)
    # viral_data = pd.DataFrame(viral_data, columns=['image', 'label'], index=None)
    # # Randomise the order
    # normal_data_list = normal_data.sample(frac=1.).reset_index(drop=True)
    # bact_data_list = bact_data.sample(frac=1.).reset_index(drop=True)
    # viral_data_list = viral_data.sample(frac=1.).reset_index(drop=True)
    # Check the data frame
    # print("val data")
    # print(f"normal:\n {normal_data_list.head()}")
    # print(f"bacterial:\n {bact_data_list.head()}")
    # print(f"viral:\n {viral_data_list.head()}")

    return normal_data_list, bact_data_list, viral_data_list

def get_test_image_data():
    # Dirs
    norm_dir = test_data_dir / 'NORMAL'
    bact_dir = test_data_dir / 'BACTERIA'
    viral_dir = test_data_dir / 'VIRUS'
    # Get the list of all the images
    normal_cases = norm_dir.glob('*.jpeg')
    bacterial_cases = bact_dir.glob('*.jpeg')
    viral_cases = viral_dir.glob('*.jpeg')

    # Initialise lists to put all the images in, along with their labels: (img, label)
    normal_data = []
    bact_data = []
    viral_data = []
    # Add images to its list and label them: No-Pneumonia: 0, Bacterial: 1, Viral: 2
    for img in normal_cases:
        normal_data.append((img, 0))
    for img in bacterial_cases:
        bact_data.append((img, 1))
    for img in viral_cases:
        viral_data.append((img, 2))


    normal_data_list = normal_data
    bact_data_list = bact_data
    viral_data_list = viral_data
    
    # # Convert to pandas data frame
    # normal_data = pd.DataFrame(normal_data, columns=['image', 'label'], index=None)
    # bact_data = pd.DataFrame(bact_data, columns=['image', 'label'], index=None)
    # viral_data = pd.DataFrame(viral_data, columns=['image', 'label'], index=None)
    # # Randomise the order
    # normal_data_list = normal_data.sample(frac=1.).reset_index(drop=True)
    # bact_data_list = bact_data.sample(frac=1.).reset_index(drop=True)
    # viral_data_list = viral_data.sample(frac=1.).reset_index(drop=True)
    # Check the data frame
    # print("test data")
    # print(f"normal:\n {normal_data_list.head()}")
    # print(f"bacterial:\n {bact_data_list.head()}")
    # print(f"viral:\n {viral_data_list.head()}")

    return normal_data_list, bact_data_list, viral_data_list

# Get lists
tr_normal_imgs_list, tr_bact_imgs_list, tr_viral_imgs_list = get_train_image_data()
val_normal_imgs_list, val_bact_imgs_list, val_viral_imgs_list  = get_val_image_data()
test_normal_imgs_list, test_bact_imgs_list, test_viral_imgs_list  = get_test_image_data()


def decode_imgs_to_data(normal_cases, bact_cases, viral_cases):

    # Initialise lists
    normal_data = []
    normal_labels = []
    bact_data = []
    bact_labels = []
    viral_data = []
    viral_labels = []

    # Normal cases
    for img in normal_cases:
        img = mimg.imread(str(img[0]))
        img = cv2.resize(img, (299,299))
        if len(img) <= 2:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        label = to_categorical(0, num_classes=2)
        normal_data.append(img)
        normal_labels.append(label)
                        
    # Bacterial cases        
    for img in bact_cases:
        img = mimg.imread(str(img[0]))
        img = cv2.resize(img, (299,299))
        if len(img) <= 2:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        label = to_categorical(0, num_classes=2)
        bact_data.append(img)
        bact_labels.append(label)

    # Viral cases        
    for img in bact_cases:
        img = mimg.imread(str(img[0]))
        img = cv2.resize(img, (299,299))
        if len(img) <= 2:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        label = to_categorical(0, num_classes=2)
        viral_data.append(img)
        viral_labels.append(label)
        
    # Convert the list into numpy arrays
    normal_data = np.array(normal_data)
    normal_labels = np.array(normal_labels)
    bact_data = np.array(bact_data)
    bact_labels = np.array(bact_labels)
    viral_data = np.array(viral_data)
    viral_labels = np.array(viral_labels)

    return (normal_data, normal_labels), (bact_data, bact_labels), (viral_data, viral_labels)


#decode_imgs_to_data(tr_normal_imgs_list, tr_bact_imgs_list, tr_viral_imgs_list)
norm, bact, viral = decode_imgs_to_data(val_normal_imgs_list, val_bact_imgs_list, val_viral_imgs_list)
#decode_imgs_to_data(test_normal_imgs_list, test_bact_imgs_list, test_viral_imgs_list)

print(f"norm:\n {norm[0][:5]}")
print(f"bact:\n {bact[0][:5]}")
print(f"viral:\n {viral[0][:5]}")



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
    # Get pretrained model
    model = applications.inception_resnet_v2.InceptionResNetV2(
                include_top=False, 
                weights='imagenet', 
                input_shape=(299, 299, 3), 
                pooling='avg'
                )
    # Freeze layers
    for layer in model.layers:
        layer.trainable = False

    # Add trainable layers to the model
    x = model.output
    #x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    # Create the final model and compile it
    final_model = Model(inputs=model.input, outputs = predictions)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    final_model.fit_generator(
                train_generator,
                steps_per_epoch = nb_train_steps,
                epochs = nb_epochs,
                validation_data = validation_generator,
                validation_steps = nb_val_steps
                #callbacks = [checkpoint, early]
                )

    return final_model

#res_model = create_model()
#res_model.summary()