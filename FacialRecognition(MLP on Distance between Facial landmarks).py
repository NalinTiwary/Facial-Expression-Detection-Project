import cv2
import dlib
import pickle
import warnings
import numpy as np
import pandas as pd

import urllib.request

from sklearn import metrics
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tqdm import tqdm,tqdm_pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import re
import gdown
import keras

from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

warnings.filterwarnings("ignore")

dataset_url = 'https://drive.google.com/uc?id=1xFiYsULlQWWmi2Ai0fHjtApniP5Pscuf'
dataset_path = './ferdata.csv'
gdown.download(dataset_url, dataset_path, True)

dlibshape_url = 'https://drive.google.com/uc?id=17D3D89Gke6i5nKOvmsbPslrGg5rVgOwg'
dlibshape_path ='./shape_predictor_68_face_landmarks.dat'
gdown.download(dlibshape_url, dlibshape_path, True)

pureX_url = 'https://drive.google.com/uc?id=1CglpXodenZVrkaZehLtfykfQv8dcnfO9'
pureX_path = './pureX.npy'
gdown.download(pureX_url, pureX_path,True)

dataX_url = 'https://drive.google.com/uc?id=1sIJGxUM6rNBcWxucs6iynDepeKU1Q56p'
dataX_path = './dataX.npy'
gdown.download(dataX_url, dataX_path, True)

dataY_url = 'https://drive.google.com/uc?id=1Rfr0OP-hZO_UZfuOyMNR2RjNRAro85zE'
dataY_path = './dataY.npy'
gdown.download(dataY_url, dataY_path, True)

def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = list(label_map.values())
  df_cm = pd.DataFrame(cm,index = labels,columns = labels)
  fig = plt.figure()
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5,3.5,4.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  plt.show()
  plt.close()

def plot_graphs(history, best):

  plt.figure(figsize=[10,4])
  plt.subplot(121)
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy across training\n best accuracy of %.02f'%best[1])
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.subplot(122)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss across training\n best accuracy of %.02f'%best[0])
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

label_map = {"0":"ANGRY","1":"HAPPY","2":"SAD","3":"SURPRISE","4":"NEUTRAL"}
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

dataX_lm = np.load('./dataX.npy')
dataY_lm = np.load('./dataY.npy')

epochs = 20
batch_size = 64
test_ratio = .1

y_onehot_lm = keras.utils.to_categorical(dataY_lm, len(set(dataY_lm)))

X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(dataX_lm, y_onehot_lm, test_size=0.1, random_state=42)

lm_scaler = StandardScaler()
lm_scaler.fit(X_train_lm)
X_train_lm = lm_scaler.transform(X_train_lm)
X_test_lm = lm_scaler.transform(X_test_lm)

pickle.dump(lm_scaler, open("lm_scaler.p", "wb"))

lm_model = Sequential()
lm_model.add(Dense(5120, activation='relu',kernel_initializer='glorot_normal', input_shape=( X_train_lm.shape[1]   ,)))
lm_model.add(Dropout(0.2))
lm_model.add(Dense(512,kernel_initializer='glorot_normal', activation='relu'))
lm_model.add(Dropout(0.2))
lm_model.add(Dense(256,kernel_initializer='glorot_normal', activation='relu'))
lm_model.add(Dropout(0.2))
lm_model.add(Dense(5, activation='softmax'))

lm_model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.001), metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_lm_model.h5', verbose=1, monitor='val_loss',save_best_only=True,  mode='auto')

lm_history = lm_model.fit(X_train_lm, y_train_lm, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test_lm, y_test_lm), shuffle=True)
lm_accuracy =lm_model.evaluate(X_test_lm, y_test_lm, batch_size=64)
plot_graphs(lm_history, lm_accuracy)

y_pred_lm = lm_model.predict_classes(X_test_lm)
y_true_lm = np.argmax(y_test_lm,axis=1)
plot_confusion_matrix(y_true_lm,y_pred_lm)
