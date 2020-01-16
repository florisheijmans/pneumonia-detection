import sys
import os
from glob import glob
from six.moves import urllib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import tensorflow as tf
import tarfile
from keras import applications

model = applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False, weights='imagenet')


model.summary()

graph_def = model.graph_def

