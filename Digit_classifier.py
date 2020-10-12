

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


saved_model = load_model('best_model.h5')


img = Image.open("baldej.png").convert('RGB')


x = np.array(img)
img = img.resize((round(x.shape[1] / (x.shape[0] / 32)), 32), Image.ANTIALIAS)
x = np.array(img)
img = img.crop((0, 0, 32 * round(x.shape[1] / 32), 32))
x = np.array(img)
plt.imshow(x)
plt.show()


for i in range(round(x.shape[1] / 32)):
    x1 = x[:, 32 * i:32 * (i + 1), :]
    x1.shape = (1, 32, 32, 3)
    prediction = saved_model.predict(x1)
    print(prediction)
