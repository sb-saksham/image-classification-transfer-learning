import os
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf

model_path = "../saksham/model"

CLASS_NAMES = {
    0: 'daisy',
    1: 'desert_cactus',
    2: 'guitar',
    3: 'ladybug',
    4: 'otter',
    5: 'panda',
    6: 'parrot',
    7: 'piano',
    8: 'sunflower',
}

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Full Path to the input images folder")
ap.add_argument("-o", "--output", required=True, help="Full Path to Output folder")
ap.add_argument("-c", "--confidence", default=80, help="Confidence value for labelling(0-100 in percentage)")

args = vars(ap.parse_args())

images_dir = args['input']
output_dir = args['output']
confidence = float(args['confidence'])/100


model = tf.keras.models.load_model(model_path, compile=False)


def classify_and_save_image(file_name):
    img_path = os.path.join(images_dir, file_name)
    image = tf.keras.utils.load_img(img_path)
    image_batch = tf.keras.utils.img_to_array(image)
    image_batch = np.array([image_batch])
    predictions = model.predict_on_batch(image_batch).flatten()
    predictions = tf.nn.sigmoid(predictions)
    label_ind = np.argmax(predictions)
    predicted_prob = predictions[label_ind]
    label = 'not_labelled'
    if predicted_prob > confidence:
        label = CLASS_NAMES.get(label_ind, 'not_labelled')
        
    final_im_path = os.path.join(output_dir, label)
    image.save(os.path.join(final_im_path, file_name))

def create_dirs(output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for label in CLASS_NAMES.values():
        label_path = os.path.join(output_dir, label)
        if not os.path.isdir(label_path):
            os.mkdir(label_path)
    not_label_path = os.path.join(output_dir, 'not_labelled')
    if not os.path.isdir(not_label_path):
        os.mkdir(not_label_path)


if __name__ == "__main__":
    create_dirs(output_dir)
    for _, _, file_names in os.walk(images_dir):
        for file_name in file_names:
            # check for extension .jpg, .png, .jpeg
            if ".jpg" in file_name or ".png" in file_name or ".jpeg" in file_name:                
                classify_and_save_image(file_name)
