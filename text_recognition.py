# Imports
import cv2
import numpy as np
import pickle
import os
import tensorflow as tf
from keras.models import load_model
from pytesseract_detection import char_list

# -------------------------------------------------------------------------------------------------------------

def text_rec(char_directory):

    # Read the contents of the pickle file
    with open('class_names.pkl', 'rb') as f:
        # Load the data from the file
        class_names = pickle.load(f)
        print(class_names)

    # Open the output file
    output_file = open("output_file.txt", "w")

    # Load the DenseNet model
    densenet_model = load_model("models/densenet_model")

    # Load the VGG19 model
    vgg19_model = load_model("models/vgg19_model")

    # Declare and initialize count variable used for tracking through predictions
    count = 0

    # Store names of files from char_imgs directory
    files = os.listdir(char_directory)
    
    # Loop through each file in the folder
    for file in files:
        # Store the image path
        img_path = os.path.join(char_directory, file)

        if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            # Read the image using OpenCV
            img = cv2.imread(img_path)
        else:
            print("Invalid image format provided. Ensure the directory contains images of type (.jpg, .jpeg, .png).")

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

        count += 1

        # Obtains predicted class and their respective probability
        for i, pred_class in enumerate(pred_classes):
            output_class = class_names[pred_class]
            output_prob = ensemble_preds[i][pred_class]
                    
            # Print the predicted character and probability 
            print(f"Predicted Character: {chr(int(output_class))}, Probability: {output_prob:.2f}")
            # Write every recognized character to the output file
            output_file.write(chr(int(output_class)))
        
        for i in char_list:
            if count == i:
                output_file.write(" ")

    # Close the output file after completion of writing
    output_file.close()

# -------------------------------------------------------------------------------------------------------------

# Main

char_directory = "char_imgs"
        
# Function call
text_rec(char_directory)

# Open and read the output file to view contents
out_file = open("output_file.txt", "r")
print("Output file contents:")
print(out_file.read())