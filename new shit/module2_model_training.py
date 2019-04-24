import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from cv2 import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
import os
import joblib
import lib


list_dir = os.listdir('dataset')
ignore = 0
if '.DS_Store' in list_dir:
    ignore = 1
    
num_classes = len(list_dir)-ignore

#carregando facenet

model_path = 'facenet_keras.h5'
model = load_model(model_path)

# Carregando dataset

#the real deal
cascade_path = r'D:\Github\FaceRecognition\new shit\haarcascade_frontalface_default.xml'
names = []
for n in range(1,num_classes):
    names.append(str(n))
image_size = 160

#treino
image_dir_basepath = 'dataset'
data = {}
y_train = []
X_train = []
for i in range(1, len(os.listdir(image_dir_basepath))):
    for k in range(len(os.listdir(os.path.join(image_dir_basepath, str(i))))):
        y_train.append(i)
for name in names:
    image_dirpath = os.path.join(image_dir_basepath, name)
    image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
    X_train.append(lib.calc_embs(image_filepaths))

#teste
image_dir_basepath = 'dataset-test'
data = {}
y_test = []
X_test = []
for i in range(1, len(os.listdir(image_dir_basepath))):
    for k in range(len(os.listdir(os.path.join(image_dir_basepath, str(i))))):
        y_test.append(i)
for name in names:
    image_dirpath = os.path.join(image_dir_basepath, name)
    image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
    X_test.append(lib.calc_embs(image_filepaths))

#prep data
X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0]* (X_train.shape[1],128)))
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0]* (X_test.shape[1],128)))

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'classifier.h5')