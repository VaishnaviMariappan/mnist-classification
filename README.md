# Convolutional Deep Neural Network for Digit Classification

## AIM
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.
## Problem Statement and Dataset

Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
## Neural Network Model

Include the neural network model diagram.

![image](https://github.com/VaishnaviMariappan/mnist-classification/assets/94169913/963d1dba-b7a1-4f02-b87a-3a370beddf10)

## DESIGN STEPS

1. Import tensorflow and preprocessing libraries.

2. Build a CNN model.

3. Compile and fit the model and then predict.
## PROGRAM
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

single_image = X_train[13]
single_image.shape
plt.imshow(single_image,cmap='gray')

y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[13]

## One hot Encoding Outputs
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape

single_image = X_train[912]
plt.imshow(single_image,cmap='gray')
y_train_onehot[912]

## Reshape Inputs
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

## Build CNN Model
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=25,kernel_size=(7,7),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(41,activation="relu"))
model.add(layers.Dense(35,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=8,
          batch_size=70,
          validation_data=(X_test_scaled,y_test_onehot))

## Metrics
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

## Prediction for a single input
img = image.load_img('Img.jpg')
type(img)

img = image.load_img('Img.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')
```

## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot:
![image](https://github.com/VaishnaviMariappan/mnist-classification/assets/94169913/e4e1039a-6bfc-403e-9acd-bf171ba7f27a)


### Accuracy, Validation Accuracy Vs Iteration:
![image](https://github.com/VaishnaviMariappan/mnist-classification/assets/94169913/02552989-7820-4359-b7f8-ecedccef2985)


### Classification Report
![image](https://github.com/VaishnaviMariappan/mnist-classification/assets/94169913/a9814a8d-4a4e-4c00-884c-401b49a5ed57)



### Confusion Matrix
![image](https://github.com/VaishnaviMariappan/mnist-classification/assets/94169913/cd8b5229-3dbb-44b4-a2b3-59072628cfe1)



### New Sample Data Prediction
![image](https://github.com/VaishnaviMariappan/mnist-classification/assets/94169913/415ebdda-c60d-49a4-b1c9-82113b4d722c)



![image](https://github.com/VaishnaviMariappan/mnist-classification/assets/94169913/e49cc428-da5d-42be-a168-cbd5eb20d965)


## RESULT

A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
