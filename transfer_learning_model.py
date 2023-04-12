from argparse import ArgumentParser
import os

import tensorflow as tf

ap = ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Full Path to the input images folder organized with directory name as labels for images in it")
ap.add_argument("-o", "--output", required=True, help="Full Path to Output folder to save the trained model")
ap.add_argument("-l", "--labels", required=True, help="The number of labels dataset is divided into")
args = vars(ap.parse_args())

train_dir = args['input']
save_dir = args['output']
try:
    LABELS = int(args['labels'])
except ValueError:
    raise Exception("Please Provide valid value for label")

BATCH_SIZE = 32
IMG_SIZE = (128, 128)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, seed=267, image_size=IMG_SIZE, validation_split=0.1, subset='training')
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, seed=267, image_size=IMG_SIZE, validation_split=0.1, subset='validation')
class_names = train_dataset.class_names

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
base_model.trainable = False
base_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
prediction_layer = tf.keras.layers.Dense(9)
prediction_batch = prediction_layer(feature_batch_average)
inputs = tf.keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()
initial_epochs = 1





loss0, accuracy0 = model.evaluate(validation_dataset)
history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)
model.save(os.path.join(save_dir,"models"))
