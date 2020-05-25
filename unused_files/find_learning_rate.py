import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from minixception import MiniXception
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

from kerastuner.tuners import RandomSearch


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

DATASET_PATH = os.path.join("datasets", "fer2013", "fer2013.csv")
IMAGE_SIZE = (48, 48)

#%%

def tune_optimizers_lr(hp):
    
    model = MiniXception((*IMAGE_SIZE, 1), 7)
    # lr = hp.Choice("learning_rate", [1e-8, 1e-6, 1e-5, 1e-4, 1e-2, 1e+1])
    lr = hp.Float("learning_rate", min_value = 1e-8, max_value = 1, sampling = "LOG")
    model.compile(optimizer = Adam(lr), loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    return model
    

#%%

data = pd.read_csv(DATASET_PATH)
pixels = data["pixels"].tolist()
width, height = IMAGE_SIZE
faces = list()
for pixel_sequence in tqdm(pixels):
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    face = cv2.resize(face.astype("uint8"), IMAGE_SIZE)
    faces.append(face.astype("float32"))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
emotions = pd.get_dummies(data["emotion"]).to_numpy()

#%%

faces = faces.astype("float32")
faces /= 255.0
faces -= 0.5
faces *= 2.0

#%%

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size = 0.2, shuffle = True)

#%%

tuner = RandomSearch(tune_optimizers_lr, objective = "val_accuracy", max_trials = 1,
                     executions_per_trial = 1,
                     directory = "test",
                     project_name = "tune_optimizers_lr")

tuner.search_space_summary()

tuner.search(x = X_train, y = y_train, epochs = 20, validation_data = (X_test, y_test), batch_size = 256)

tuner.results_summary()
