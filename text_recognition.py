# Imports
# import tensorflow as tf
# import numpy as np
import pickle
from keras.models import load_model

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


# Main
        
# Function calls
text_recognition()