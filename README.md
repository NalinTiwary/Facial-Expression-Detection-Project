# Facial-Emotion-Detection-Project
This was a Project I completed during my Summer Vacation to get a better understanding of Neural Networks.This repository contains code that uses Deep Learning to train a model to predict the emotion a person is showing from his/her Facial Expressions. The model can predict whether the person is Sad, Happy, Angry, Suprised or Neutral. I have trained both a Multi-Layer Perceptron and a Convolutional Neural Network to make the prediction. I have also used Transferred Learning to make a prediction using the VGG model. The camera.py file can be used to activate the computer's web-camera and use the models real time.


## Keras
Keras is a python library used commonly to create and use various Machine Learning Models in Python.

### Installation 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install keras.
```bash
pip install keras
```
### Usage

```python
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.applications.vgg16 import VGG16 #Used to import a pre-existing model for transffered learning

X_train, X_test, y_train, y_test = train_test_split(dataX_pixels, y_onehot, test_size=test_ratio, random_state=42) #Used to split data into  training and testing data in a particular ration(mostly 80 to 20(4:1))

cnn_model = Sequential()  #Ensures that the model goes from left to right
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)))  #Adding 2 convolutional layers to the model which both take the image directly as input
cnn_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(width, height, 1)))
cnn_model.add(BatchNormalization()) # Normalizing the input layer by modifying activations to increase effectivness
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
cnn_model.add(Dropout(0.5))  #Setting Dropout to 0.5 to prevent overfitting
cnn_model.add(Flatten())  #Changing the input shape to only 1 dimension
cnn_model.add(Dense(512, activation='relu'))  #Adding Dense layers to the model just like a regular MLP
cnn_model.add(Dropout(0.4))

checkpoint = ModelCheckpoint('best_cnn_model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  #Used to find the best model in terms of loss

cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) #Creating the CNN with the optimizer 'Adam' and evaluation metric of accuracy
cnn_history = cnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test), shuffle=True) #Stores the best model
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
