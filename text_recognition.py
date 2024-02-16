# Imports
import cv2
import numpy as np
import pickle
import tensorflow as tf
from keras.models import load_model

# Text recognition function
def text_rec():

    # Read the contents of the pickle file
    with open('class_names.pkl', 'rb') as f:
        # Load the data from the file
        class_names = pickle.load(f)
        print(class_names)

    # Load the DenseNet model
    densenet_model = load_model("models/densenet_model")

    # Load the VGG19 model
    vgg19_model = load_model("models/vgg19_model")

    # # Confidence threshold
    # confidence_threshold = 0.70

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
    # Use DenseNet model to make a prediction on the normalized image
    densenet_pred = densenet_model(normalized_img)

    # Use VGG19 model to make a prediction on the normalized image
    vgg19_pred = vgg19_model(normalized_img)

    # Compute the average prediction of the two models
    ensemble_preds = np.mean([densenet_pred, vgg19_pred], axis=0)

    # Get the predicted classes for each image
    pred_classes = np.argmax(ensemble_preds, axis=1)

    # Obtains predicted class and their respective probability
    for i, pred_class in enumerate(pred_classes):
        output_class = class_names[pred_class]
        output_prob = ensemble_preds[i][pred_class]
                    
        # Print the predicted character and probability 
        print(f"Predicted Character: {chr(int(output_class))}, Probability: {output_prob:.2f}")

# -------------------------------------------------------------------------------------------------------------

# Main
        
# Function calls
text_rec()