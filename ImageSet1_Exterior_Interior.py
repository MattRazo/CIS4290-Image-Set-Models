# all the imports we most likely need

import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from pathlib import Path
from tensorflow import keras
from tensorflow import compat
from tensorflow import version
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import optimizers


# import zipfile
# with zipfile.ZipFile("Train_9G.zip") as zip_ref:
#     zip_ref.extractall("targetdir")

dataset = pathlib.Path(r'H:\Downloads\Train2')
print(dataset)
image_count = len(list(dataset.glob('*/*.jpg')))
print("image count " + str(image_count))

# an array(the exterior folder) which will hold the images inside the folder?


batch_size = 32
img_height = 200
img_width = 200

# train
# 20% for validation? if there's 149994 images in total or that meets criteria. 80% for training. 20% validation = 29087

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


for image_batch, labels_batch in val_ds:
    print("val_ds images")
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# end of displaying images for val_ds
AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# train_ds = train_ds.take(1000).cache().repeat()
# val_ds = val_ds.take(1000).cache().repeat()
val_size = int(image_count*0.2)
print("val dataset size " + str(val_size))
train_size = int(image_count*0.8)
print("train dataset size " + str(train_size))
train_ds = train_ds.take(250).cache()
val_ds = val_ds.take(400).cache()  #originally 250

# train_ds = train_ds.padded_batch(batch_size)
# train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

data_augmentation = Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

num_classes = 2

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.compile(optimizer=tf.keras.optimizer.Adam(learning_rate=1e-3),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=[tf.keras.metrics.BinaryAccuracy(),
#                        tf.keras.metrics.FalseNegatives()])

model.summary()

epochs = 30
history = model.fit(train_ds,
                    validation_data=val_ds,
                    batch_size=batch_size,
                    epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

