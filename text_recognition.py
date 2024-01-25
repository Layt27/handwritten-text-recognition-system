# Imports
# import tensorflow as tf
# import numpy as np
import pickle
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import easyocr

# -------------------------------------------------------------------------------------------------------------

def text_recognition():

    # Read the contents of the pickle file
    with open('class_names.pkl', 'rb') as f:
        # Load the data from the file
        class_names = pickle.load(f)
    
    print("Printing the contents of the pickle file: \n", class_names)

    # # Confidence threshold
    # confidence_threshold = 0.85

    # Load model 1
    # model1 = load_model("models/densenet_model")

    # # Load model 2
    # model2 = tf.saved_model.load("models/efficientnetb0_model")

    # Code for showing bounding boxes on letters
    
    # Load image using OpenCV
    img_path = "test_imgs/test_img_9.jpg"
    img = cv2.imread(img_path)

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    try:
        # Use EasyOCR to perform OCR and get the coordinates of the bounding boxes
        results = reader.readtext(img)
        # print(results)
    except Exception as e:
        print(f"Error during OCR: {e}")
        return


    # # Draw bounding boxes on the image
    # for (bbox, text, prob) in results:
    #     (top_left, top_right, bottom_right, bottom_left) = bbox
    #     top_left = tuple(map(int, top_left))
    #     bottom_right = tuple(map(int, bottom_right))
    #     cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # # Display the image with bounding boxes
    # plt.imshow(img)
    # plt.title("Detected Characters")
    # plt.show()


    # Extract each bounding box based on the detected bounding boxes
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x, y, w, h = int(top_left[0]), int(top_left[1]), int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
        each_bbox = img[y:y + h, x:x + w]

        # Display each bounding box
        plt.imshow(each_bbox)
        plt.title("Detected Characters")
        plt.show()

# -------------------------------------------------------------------------------------------------------------

# Main
        
# Function calls
text_recognition()