

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pylab as plt
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model


def img_to_array(img_name):
    img = Image.open(
        "eaSF/all_letters_image/all_letters_image/" + img_name).convert('RGB')
    img = img.crop((0, 0, 32, 32))
    x = np.array(img)
    return np.expand_dims(x, axis=0)


def data_to_tensor(img_names):
    list_of_tensors = [img_to_array(img_name) for img_name in img_names]
    return np.vstack(list_of_tensors)


data = pd.read_csv("all_letters_info.csv")
image_names = data['file']
letters = data['letter']
targets = data['label'].values
tensors = data_to_tensor(image_names)


X = tensors.astype("float32") / 255
y = targets
y = keras.utils.to_categorical(y - 1, 33)


X_train_whole, X_test, y_train_whole, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_whole, y_train_whole, test_size=0.1, random_state=1)


print(X_train.shape)
datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
print(datagen.fit(X_train))


deep_RU_model = Sequential()

deep_RU_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(32, 32, 3)))
deep_RU_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
deep_RU_model.add(MaxPooling2D(pool_size=(2, 2)))
deep_RU_model.add(Dropout(0.25))


deep_RU_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
deep_RU_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
deep_RU_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
deep_RU_model.add(Dropout(0.25))


deep_RU_model.add(Flatten())
deep_RU_model.add(Dense(256, activation="relu"))
deep_RU_model.add(Dropout(0.3))
deep_RU_model.add(Dense(33, activation="softmax"))


deep_RU_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(
    lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=["accuracy"])
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                     mode='max', verbose=1, save_best_only=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0000001)
history = deep_RU_model.fit(datagen.flow(X_train, y_train, batch_size=90), validation_data=(
    X_val, y_val), epochs=139, callbacks=[learning_rate_reduction, es, mc])


saved_model = load_model('best_model.h5')
_, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
print(test_acc)
