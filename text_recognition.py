# Imports
import cv2
import numpy as np
import pickle
import tensorflow as tf
from keras.models import load_model

# Text recognition function
def text_rec(char_directory):

    # Read the contents of the pickle file
    with open('class_names.pkl', 'rb') as f:
        # Load the data from the file
        class_names = pickle.load(f)
        print(class_names)

    # Load the DenseNet model
    densenet_model = load_model("models/densenet_model")

    # # Load the VGG19 model
    # vgg19_model = load_model("models/vgg19_model")

    img_path = "char_imgs/char_181.jpg"
    img = cv2.imread(img_path)

    # Maintain aspect ratio of image
    h, w, _ = img.shape
    scale = max(h, w) / 64
    new_h = int(h / scale)
    new_w = int(w / scale)
    top = (64 - new_h) // 2
    bottom = 64 - new_h - top
    left = (64 - new_w) // 2
    right = 64 - new_w - left

    # Resize the image to match the input shape expected by the model
    resized_img = cv2.resize(img, (new_w, new_h))
    resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Normalize image
    normalized_img = tf.keras.preprocessing.image.img_to_array(resized_img)
    normalized_img = np.expand_dims(normalized_img, axis=0)
    normalized_img = tf.cast(normalized_img, dtype=tf.float32)

    # Make predictions
    predictions = densenet_model.predict(normalized_img)

    # Get the predicted class for the character
    predicted_class = np.argmax(predictions, axis = 1)

    # Obtains predicted class and their respective probability
    for i, pred_class in enumerate(predicted_class):
        output_class = class_names[pred_class]
        output_prob = predictions[i][pred_class]

        print(f"Character {i+1}: {output_class}, Probability: {output_prob:.2f}")

# -------------------------------------------------------------------------------------------------------------

# Main
    
char_directory = "char_imgs"
        
# Function calls
text_rec(char_directory)