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
from keras.models import Sequential, Model, load_model
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
from mlxtend.evaluate import mcnemar_table, mcnemar



# Visualisation
from gradcamutils import GradCam, GradCamPlusPlus, ScoreCam, build_guided_model, GuidedBackPropagation, superimpose, read_and_preprocess_img

# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'
# Set the numpy seed
np.random.seed(111)
# Disable multi-threading in tensorflow ops
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# Set the random seed in tensorflow at graph level
tf.compat.v1.set_random_seed(111)
# Define a tensorflow session with above session configs
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# Set the session in keras
tf.compat.v1.keras.backend.set_session(sess)
# Make the augmentation sequence deterministic
aug.seed(111)

total_train_imgs = 5098
total_val_imgs = 519

cwd = os.getcwd()
normpneum_bin_file_dir = os.path.join(cwd, "chest_xray", "decoded_imgs", "normal_pneum")
bactviral_bin_file_dir = os.path.join(cwd, "chest_xray", "decoded_imgs", "bact_viral") 

# Define hyperparameters
img_width, img_height = 299, 299
train_data_dir = Path("chest_xray/images/train")
val_data_dir = Path("chest_xray/images/val")
test_data_dir = Path("chest_xray/images/test")
normpneum_batch_size = 16
bactviral_batch_size = 16
nb_epochs = 20
normpneum_nb_train_steps = total_train_imgs / nb_epochs
bactviral_nb_train_steps = total_train_imgs / nb_epochs
nb_val_steps = total_val_imgs


def get_image_data():
    # Globalise variables
    global normpneum_test_data; global normpneum_test_labels
    global normpneum_val_data; global normpneum_val_labels
    global normpneum_test_data; global normpneum_test_labels

    # Get normal-pneumonia lists
    try:
        normpneum_val_data, normpneum_val_labels, normpneum_test_data, normpneum_test_labels = load_numpy_binary(normpneum_bin_file_dir)
        print("Try accepted: Loaded normal-pneumonial validation and test data")
    except:
    print("Except: Getting all normal-pneumonial image lists")
    normpneum_val_dat, bactviral_val_dat = create_image_data(val_data_dir)
    normpneum_test_dat, bactviral_test_dat = create_image_data(test_data_dir)

    normpneum_val_data, normpneum_val_labels = decode_imgs_to_data(normpneum_val_dat)
    normpneum_test_data, normpneum_test_labels = decode_imgs_to_data(normpneum_test_dat)

    create_all_binary_files(
                            normpneum_val_data, 
                            normpneum_val_labels, 
                            normpneum_test_data, 
                            normpneum_test_labels,
                            normpneum_bin_file_dir
                            )

    # Globalise variables
    global bactviral_test_data; global bactviral_test_labels
    global bactviral_val_data; global bactviral_val_labels
    global bactviral_test_data; global bactviral_test_labels

    # Get bacterial-viral lists
    try:
        bactviral_val_data, bactviral_val_labels, bactviral_test_data, bactviral_test_labels = load_numpy_binary(bactviral_bin_file_dir)
        print("Try accepted: Loaded bacterial-viral validation and test data")
    except:
    print("Except: Getting all bacterial-viral image lists")
    normpneum_val_dat, bactviral_val_dat = create_image_data(val_data_dir)
    normpneum_test_dat, bactviral_test_dat  = create_image_data(test_data_dir)

    bactviral_val_data, bactviral_val_labels = decode_imgs_to_data(bactviral_val_dat)
    bactviral_test_data, bactviral_test_labels = decode_imgs_to_data(bactviral_test_dat)

    create_all_binary_files(
                            bactviral_val_data, 
                            bactviral_val_labels, 
                            bactviral_test_data, 
                            bactviral_test_labels,
                            bactviral_bin_file_dir
                            )

    # Read training data
    normpneum_train_dat, bactviral_train_dat = create_image_data(train_data_dir)
    # Convert to pandas data frame
    normpneum_train_data = pd.DataFrame(normpneum_train_dat, columns=['image', 'label'], index=None)
    bactviral_train_data = pd.DataFrame(bactviral_train_dat, columns=['image', 'label'], index=None)

    # Get a train data generators
    global normpneum_train_data_gen
    normpneum_train_data_gen = data_gen(data=normpneum_train_data, batch_size=normpneum_batch_size)
    global bactviral_train_data_gen
    bactviral_train_data_gen = data_gen(data=bactviral_train_data, batch_size=bactviral_batch_size)

    # Define the number of training steps
    normpneum_nb_train_steps = normpneum_train_data.shape[0]//normpneum_batch_size
    bactviral_nb_train_steps = bactviral_train_data.shape[0]//bactviral_batch_size


def create_image_data(data_dir):
    # Dirs
    norm_dir = data_dir / 'NORMAL'
    bact_dir = data_dir / 'BACTERIA'
    viral_dir = data_dir / 'VIRUS'
    # Get the list of all the images
    normal_cases = norm_dir.glob('*.jpeg')
    bacterial_cases = bact_dir.glob('*.jpeg')
    viral_cases = viral_dir.glob('*.jpeg')

    # Initialise lists to put all the images in, along with their labels: (img, label)
    norm_pneum_dat = []
    bact_viral_dat = []
    # Add images to its list and label them: Normal: 0, and Pneumonia: 1
    for img in normal_cases:
        norm_pneum_dat.append((img, 0))
    for img in bacterial_cases:
        norm_pneum_dat.append((img, 1))
        bact_viral_dat.append((img, 0))
    for img in viral_cases:
        norm_pneum_dat.append((img, 1))
        bact_viral_dat.append((img, 1))

    print("Got all image data from", data_dir)
    return norm_pneum_dat, bact_viral_dat


def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 299, 299, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)

    # Augmentation sequence
    seq = iaa.OneOf([
        iaa.Fliplr(), # horizontal flips
        iaa.Affine(rotate=20), # roatation
        iaa.Multiply((1.2, 1.5))]) #random brightness

    # Initialize a counter
    i = 0
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
            img_dat = mimg.imread(str(img_name))
            img_dat = cv2.resize(img_dat, (299,299))
            if len(img_dat) <= 2:
                img_dat = np.dstack([img_dat, img_dat, img_dat])
            else:
                img_dat = cv2.cvtColor(img_dat, cv2.COLOR_BGR2RGB)

            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img_dat, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            # orig_img = img_dat.astype(np.float32)/255.
            # img_dat = tf.subtract(img_dat, input_mean)
            # img_dat = tf.multiply(img_dat, 1.0 / input_std)
            orig_img = (img_dat.astype(np.float32) - 128) * (1/128)
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label

            # generating more samples of the undersampled class
            if label==0 and count < batch_size-2:
                aug_img1 = seq.augment_image(img_dat)
                aug_img2 = seq.augment_image(img_dat)
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
        yield batch_data, batch_labels

        if i>=steps:
            i=0


def decode_imgs_to_data(cases):
    # Initialise lists
    dat = []
    labels = []
    # Append all images to dat and labels
    counter = 0

    input_mean=128
    input_std=128

    for img in cases:
        label = to_categorical(img[1], num_classes=2)
        img = mimg.imread(str(img[0]))
        img = cv2.resize(img, (299,299))
        if len(img) <= 2:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 128) * (1/128)
        # img = tf.subtract(img, input_mean)
        # img = tf.multiply(img, 1.0 / input_std)
        dat.append(img)
        labels.append(label)
        counter += 1
        if (counter % 100 == 0):
            print(counter, "from class", label)

    # Convert the list into numpy arrays
    dat = np.array(dat)
    labels = np.array(labels)

    return dat, labels


def create_numpy_binary(np_arr, file_path, file_name):
    print(f"Creating binary file for: {file_name}")
    bin_file_name = file_name

    # Create .npy-file
    res_path = os.path.join(file_path, bin_file_name)
    np.save(res_path, np_arr)
    print(res_path)


def create_all_binary_files(val_data, val_labels, test_data, test_labels, bin_file_dir):
    create_numpy_binary(val_data, bin_file_dir, 'VALIDATION_DATA_set')
    create_numpy_binary(val_labels, bin_file_dir, 'VALIDATION_LABELS_set')
    create_numpy_binary(test_data, bin_file_dir, 'TEST_DATA_set')
    create_numpy_binary(test_labels, bin_file_dir, 'TEST_LABELS_set')


def load_numpy_binary(file_path):
    file_path = Path(file_path)

    # Get the list of all the images
    for npy_file in file_path.rglob('*.npy'):
        str_npy_file = str(npy_file)
        if 'TRAIN' in str_npy_file:
            if 'DATA' in str_npy_file:
                train_dat = np.load(npy_file, mmap_mode=None, allow_pickle=True, fix_imports=True)
            elif 'LABELS' in str_npy_file:
                train_labels = np.load(npy_file, mmap_mode=None, allow_pickle=True, fix_imports=True)
            else:
                print("Train data file doesn't mention data type")
        elif 'VALIDATION' in str_npy_file:
            if 'DATA' in str_npy_file:
                val_dat = np.load(npy_file, mmap_mode=None, allow_pickle=True, fix_imports=True)
            elif 'LABELS' in str_npy_file:
                val_labels = np.load(npy_file, mmap_mode=None, allow_pickle=True, fix_imports=True)
            else:
                print("Validation data file doesn't mention data type")
        elif 'TEST' in str_npy_file:
            if 'DATA' in str_npy_file:
                test_dat = np.load(npy_file, mmap_mode=None, allow_pickle=True, fix_imports=True)
            elif 'LABELS' in str_npy_file:
                test_labels = np.load(npy_file, mmap_mode=None, allow_pickle=True, fix_imports=True)
            else:
                print("Test data file doesn't mention data type")

    # return train_dat, train_labels, val_dat, val_labels, test_dat, test_labels
    # print(val_dat[0])
    return val_dat, val_labels, test_dat, test_labels

def create_empty_model():
    model = applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False, #Default:(299,299,3)
        weights='imagenet',
        input_shape=(299,299,3),
        pooling='max'
    )

    # Freeze layers
    for layer in model.layers:
        layer.trainable = False

    # Add trainable layers to the model
    x = model.output
    #model.summary()
    predictions = Dense(2, activation='softmax')(x)

    # Create the final model and compile it
    final_model = Model(inputs=model.input, outputs=predictions)

    # Compile model with optimization setting
    opt = Adam(lr=0.001, decay=1e-5)
    final_model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

    return final_model

def create_model(train_data_generator, val_data, val_labels, chkpt, nb_train_steps):
    print("Start creating model")
    # Get pretrained model
    model = create_empty_model()

    # More optimization of model training
    es = EarlyStopping(patience=5)

    # Fit the model
    model.fit_generator(
        train_data_generator,
        steps_per_epoch = nb_train_steps,
        epochs = nb_epochs,
        validation_data = (val_data, val_labels),
        callbacks=[es, chkpt]
    )

    return model


def create_test_model():
    # Example of testing bact vs viral
    test_model = create_empty_model()
    test_model.load_weights('best_bactviral_checkpoint.hdf5')

    loss, acc = test_model.evaluate(bactviral_test_data,  bactviral_test_labels, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

    # Get predictions
    preds = test_model.predict(bactviral_test_data, batch_size=16)
    preds = np.argmax(preds, axis=-1)

    # Original labels
    orig_test_labels = np.argmax(bactviral_test_labels, axis=-1)

    cm  = confusion_matrix(orig_test_labels, preds)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Bacterial', 'Viral'], fontsize=16)
    plt.yticks(range(2), ['Bacterial', 'Viral'], fontsize=16)
    plt.show()


def train_normpneum_model():
    checkpoint = ModelCheckpoint(filepath='best_normpneum_newrange_checkpoint.hdf5', save_best_only=True, save_weights_only=True)
    
    return create_model(
        normpneum_train_data_gen,
        normpneum_val_data, normpneum_val_labels, 
        checkpoint, normpneum_nb_train_steps
    )

def train_bactviral_model():
    checkpoint = ModelCheckpoint(filepath='best_bactviral_newrange_checkpoint.hdf5', save_best_only=True, save_weights_only=True)
    
    return create_model(
        bactviral_train_data_gen,
        bactviral_val_data, bactviral_val_labels, 
        checkpoint, bactviral_nb_train_steps
    )


get_image_data()

# Create & save best models
normpneum_model = train_normpneum_model()
# normpneum_model.save('incv3_normpneum_model.h5')

bactviral_model = train_bactviral_model()
# bactviral_model.save('incv3_bactviral_model.h5')

# normpneum_test_model = load_model('incresnetv2_normpneum_model.h5')
# bactviral_test_model = load_model('incresnetv2_bactviral_model.h5')


# def combined_classify():
#     #for case in normpneum_test_data:
#     normpneum_class_preds = normpneum_test_model.predict_classes(normpneum_test_data, batch_size=normpneum_batch_size)
#     # Save results in .csv-file
#     csv_path = normpneum_bin_file_dir + '.csv'
#     normpneum_preds_df = pd.DataFrame(normpneum_class_preds)
#     normpneum_preds_df.to_csv(csv_path)

#     # Select pneumonial cases for next model
#     pneum_indices = np.where(normpneum_class_preds == 1)
#     pneum_cases = np.take(normpneum_test_data, pneum_indices)
# get_image_data()

# normpneum_test_model = create_empty_model()
# normpneum_test_model.load_weights('best_normpneum_checkpoint.hdf5')
# bactviral_test_model = create_empty_model()
# bactviral_test_model.load_weights('best_bactviral_checkpoint.hdf5')

# def combined_classify_to_csv():
#     # Get predictions of normal-pneumonia model
#     try:
#         csv_path = normpneum_bin_file_dir + '.csv'
#         normpneum_res = pd.read_csv(csv_path).to_numpy()
#         normpneum_class_preds = normpneum_res[:,3]
#         print("Try succeeded: normal-pneumonia read from .csv-file")
#     except:
#         print("Except started: Start predicting Normal-Pneumonia cases")
#         normpneum_class_probs = normpneum_test_model.predict(normpneum_test_data, batch_size=normpneum_batch_size)
#         normpneum_class_preds = np.argmax(normpneum_class_probs, axis=-1)
#         # Save results in .csv-file
#         csv_path = normpneum_bin_file_dir + '.csv'
#         normpneum_probs_df = pd.DataFrame(normpneum_class_probs)
#         normpneum_preds_df = pd.DataFrame(normpneum_class_preds)
#         normpneum_res_df = pd.concat([normpneum_probs_df.reset_index(drop=True), normpneum_preds_df], axis=1)
#         normpneum_res_df.to_csv(csv_path, header=False)

#     # Select pneumonial cases for next model
#     pneum_indices = np.where(normpneum_class_preds == 1)
#     #pneum_cases = np.take(normpneum_test_data, pneum_indices)
#     pneum_cases = np.array([normpneum_test_data[i] for i in pneum_indices])[0]

#     # Get predictions of bacterial-viral model
#     try:
#         csv_path = bactviral_bin_file_dir + '.csv'
#         bactviral_res = pd.read_csv(csv_path).to_numpy()
#         bactviral_class_preds = bactviral_res[:,3]
#         print("Try succeeded: Bacterial-Viral read from .csv-file")
#     except:
#         print("Except started: Start predicting Bacterial-Viral cases")
#         bactviral_class_probs = bactviral_test_model.predict(pneum_cases, batch_size=bactviral_batch_size)
#         bactviral_class_preds = np.argmax(bactviral_class_probs, axis=-1)
#         # Save results in .csv-file
#         csv_path = bactviral_bin_file_dir + '.csv'
#         bactviral_probs_df = pd.DataFrame(bactviral_class_probs)
#         bactviral_preds_df = pd.DataFrame(bactviral_class_preds)
#         bactviral_res_df = pd.concat([bactviral_probs_df.reset_index(drop=True), bactviral_preds_df], axis=1)
#         bactviral_res_df.to_csv(csv_path)

#     # Predict selected cases with bacterial-viral model
#     bactviral_class_preds = bactviral_test_model(pneum_cases, batch_size=bactviral_batch_size)
#     # Save results in .csv-file
#     csv_path = bactviral_bin_file_dir + '.csv'
#     bactviral_preds_df = pd.DataFrame(bactviral_class_preds)
#     bactviral_preds_df.to_csv(csv_path)
    

# combined_classify()

# new_model = tf.keras.models.load_model('')

# # Check its architecture
# new_model.summary()


# Example of testing bact vs viral
# test_model = create_empty_model()
# # test_model.summary()

# test_model.load_weights('best_normpneum_checkpoint.hdf5')

# layer_name = 'block8_10_conv'
# img_array = read_and_preprocess_img('D:\Studie\Git-repos\pneumonia-babies\chest_xray\images\\train\BACTERIA\BACTERIA-558657-0001.jpeg', size=(299,299))

# score_cam = ScoreCam(test_model,img_array,layer_name)
# # score_cam_superimposed = superimpose('D:\Studie\Git-repos\pneumonia-babies\chest_xray\images\\train\BACTERIA\BACTERIA-558657-0001.jpeg', score_cam)

# plt.imshow(score_cam)
# # plt.imshow(score_cam_superimposed)
# plt.show()

# loss, acc = test_model.evaluate(normpneum_test_data,  normpneum_test_labels, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

# # Get predictions
# preds = test_model.predict(normpneum_test_data, batch_size=16)
# preds = np.argmax(preds, axis=-1)

# # Original labels
# orig_test_labels = np.argmax(normpneum_test_labels, axis=-1)

# cm  = confusion_matrix(orig_test_labels, preds)
# plt.figure()
# plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
# plt.xticks(range(2), ['Bacterial', 'Viral'], fontsize=16)
# plt.yticks(range(2), ['Bacterial', 'Viral'], fontsize=16)
# plt.show()
#combined_classify_to_csv()



# def statistics():
#     y_model1 = np.array([1,1,1,1,1,1,1])
#     y_model2 = np.array([1,1,1,1,1,1,1])
#     y_target = np.array([1,1,0,1,0,1,0])

    
#     tb = mcnemar_table(y_target=y_target, 
#                        y_model1=y_model1, 
#                        y_model2=y_model2)

#     print (tb)
#     tb_b = np.array([[9945, 25],
#                  [15, 15]])

#     chi2, p = mcnemar(ary=tb_b, corrected=True)
#     print('chi-squared:', chi2)
#     print('p-value:', p)




# statistics()
