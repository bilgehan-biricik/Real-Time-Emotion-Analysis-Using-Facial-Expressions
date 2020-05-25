# %% Imports

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from minixception import MiniXception

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% Limiting GPU memory growth

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

# %% Constants

LABELS = ["angry", "disgust", "fear", "happy", "sad", "suprise", "neutral"]
DATASET_PATH = os.path.join("datasets", "fer2013", "fer2013.csv")
TRAINED_FER_MODELS_PATH = os.path.join("models", "trained_fer_models")
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 256
NUM_OF_EPOCHS = 1000
PATIENCE = 100

# %% Load FER2013 dataset

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

# %% Preprocessing

faces = faces.astype("float32")
faces /= 255.0
faces -= 0.5
faces *= 2.0

# %% Overview dataset

fig = plt.figure(figsize=(48, 48))
for i in range(0, 300):
    input_img = faces[i:(i+1), :, :, :]
    ax = fig.add_subplot(16, 16, (i+1))
    ax.imshow(input_img[0, :, :, 0], cmap=plt.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt.show()

# %% Split dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(
    faces, emotions, test_size=0.2, shuffle=True)

# %% Define an image data generator

data_generator = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True)

# %% Load model and compile

model = MiniXception((*IMAGE_SIZE, 1), 7)
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
# model.summary()
plot_model(model, to_file="images/model_plot.png", show_shapes=True, show_layer_names=True)

# %% Callbacks

model_checkpoint = ModelCheckpoint(TRAINED_FER_MODELS_PATH + "/mini_xception.{val_accuracy:.2f}-{epoch:02d}.hdf5",
                                   monitor="val_accuracy",
                                   mode="max",
                                   verbose=1,
                                   save_best_only=True)
early_stop = EarlyStopping("val_loss", patience=PATIENCE)
reduce_lr = ReduceLROnPlateau(
    "val_loss", factor=0.2, patience=PATIENCE // 4, verbose=1)
callbacks = [model_checkpoint, early_stop, reduce_lr]

# %% Evaluating model

H = model.fit_generator(data_generator.flow(X_train, y_train, BATCH_SIZE),
                        steps_per_epoch=len(X_train) // BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        epochs=NUM_OF_EPOCHS,
                        callbacks=callbacks,
                        verbose=1)

# %% Make predictions on test set

predictions = model.predict(X_test, batch_size=BATCH_SIZE)
print(confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1)))

# %% Plotting accuracy and loss

N = np.arange(0, early_stop.stopped_epoch + 1)
plt.style.use("ggplot")

plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("images/loss.png", dpi=300)

plt.figure()
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("images/acc.png", dpi=300)
