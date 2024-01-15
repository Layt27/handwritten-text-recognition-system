# Imports
# import tensorflow as tf
# import numpy as np
# import pickle
# from keras.models import load_model
import matplotlib.pyplot as plt
import cv2

# -------------------------------------------------------------------------------------------------------------

def text_recognition():

    # # Read the contents of the pickle file
    # with open('class_names.pkl', 'rb') as f:
    #     # Load the data from the file
    #     class_names = pickle.load(f)

    # # Confidence threshold
    # confidence_threshold = 0.85

    # # Load model 1
    # model1 = load_model("models/vgg19_model")

    # # Load model 2
    # model2 = tf.saved_model.load("models/efficientnetb0_model")


    # Store the test image path
    test_img_path = "test_imgs/test_img_1.jpg"

    if test_img_path.lower().endswith((".jpg", ".jpeg", ".png")):
        print("Image is valid.")

        # Read the image using OpenCV
        test_img = cv2.imread(test_img_path)

        # Convert the image to grayscale
        grayscale_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        plt.imshow(grayscale_test_img, cmap = "gray")        # Displays/generates the image
        plt.axis("off")         # Removes axis labels
        plt.show()          # Renders and displays the plot containing the image

    else:
        print("Invalid image format provided.")



text_recognition()