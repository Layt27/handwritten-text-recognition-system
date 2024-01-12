# Imports
import os
from PIL import Image
import numpy as np
from datetime import datetime

# Functions

# Invert colors of images function
def invert_colors(directory):
    start = datetime.now()
    print("Attempting to invert color of images...")

    folders = os.listdir(directory)

    # Loop through each folder in the directory
    for folder in folders:
        folder_path = os.path.join(directory, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            # Loop through each file in the folder
            for file in files:
                # Construct the full path to the image
                image_path = os.path.join(folder_path, file)
                # print(image_path)

                # Check if the file is an image file
                if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                  # Open the image
                  img = Image.open(image_path)

                  # Convert the image to a NumPy array
                  img_array = np.array(img)

                  # Invert the colors (black to white, white to black)
                  inverted_img_array = np.invert(img_array)

                  # Convert the NumPy array back to an image
                  inverted_img = Image.fromarray(inverted_img_array)

                  # Save the inverted image
                  inverted_img.save(image_path)

  
    duration = datetime.now() - start
    print("Time taken to invert images: ", duration)


# Code
    
# Set the directory path containing the images
directory = "C:/Users/layto/Desktop/handwritten_to_digital_text/ascii_dataset"

# Function calls
invert_colors(directory)
