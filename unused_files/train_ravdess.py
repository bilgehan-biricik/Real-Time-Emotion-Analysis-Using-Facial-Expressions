import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

#%%

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
#%%

LABELS = {"neutral": 0, "calm": 1, "happy": 2, "sad": 3, "angry": 4, "fearful": 5, "disgust": 6, "suprised": 7}
DATASET_PATH = os.path.join("datasets", "RAVDESS")
TRAINED_FER_MODELS_PATH = os.path.join("models", "trained_fer_models")
NUM_OF_EPOCHS = 1000
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 1
PATIENCE = 50

#%%

faces = list()
emotions = list()

for imagePath in tqdm(os.listdir(DATASET_PATH)):
    emotion = LABELS[imagePath]
    images = os.listdir(os.path.join(DATASET_PATH, imagePath))
    for image in images:
        img = cv2.imread(os.path.join(DATASET_PATH, imagePath, image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        faces.append(img)
        emotions.append(emotion)
        
faces = np.array(faces)
emotions = np.array(emotions)

#%%

lb = LabelBinarizer()
emotions = lb.fit_transform(emotions)

#%%

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size = 0.2, shuffle = True) 

#%%

data_generator = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 30, 
                                    zoom_range = 0.15,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.15,
                                    horizontal_flip = True,
                                    fill_mode = "nearest")

mean = np.array([123.68, 116.779, 103.939], dtype = "float32")
data_generator.mean = mean

#%%

base_model = ResNet50(weights = "imagenet", include_top = False, input_tensor = Input(shape = (*IMAGE_SIZE, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size = (7, 7))(head_model)
head_model = Flatten(name = "flatten")(head_model)
head_model = Dense(512, activation = "relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(len(lb.classes_), activation = "softmax")(head_model)

model = Model(inputs = base_model.input, outputs = head_model)

model.compile(loss = "categorical_crossentropy", 
              optimizer = SGD(learning_rate = 1e-5, momentum = 0.9), 
              metrics = ["accuracy"])

#%%

for layer in base_model.layers:
 	layer.trainable = False

#%%

model_checkpoint = ModelCheckpoint(TRAINED_FER_MODELS_PATH + "/resnet50.{val_accuracy:.2f}-{epoch:02d}.hdf5",
                                   monitor = "val_accuracy",
                                   mode = "max",
                                   verbose = 1, 
                                   save_best_only = True)
early_stop = EarlyStopping("val_loss", patience = PATIENCE)
reduce_lr = ReduceLROnPlateau("val_loss", factor = 0.1, patience = PATIENCE // 4, verbose = 1)
callbacks = [model_checkpoint, early_stop, reduce_lr]

#%%

H = model.fit_generator(data_generator.flow(X_train, y_train, BATCH_SIZE),
                        steps_per_epoch = len(X_train) // BATCH_SIZE,
                        validation_data = (X_test, y_test),
                        epochs = NUM_OF_EPOCHS,
                        callbacks = callbacks,
                        verbose = 1,
                        shuffle = True)
