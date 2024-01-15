# Imports
import os
import cv2
from datetime import datetime

# Functions

# Conversion and inversion of images function
def convert_invert(directory):
    start = datetime.now()
    print("Commencing conversion and inversion of images...")

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
                  # Read the image using OpenCV
                  img = cv2.imread(image_path)

                  # Convert the image to grayscale
                  grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                  # Invert the colors of the image (reverse intensity values)
                  inverted_grayscale_img = cv2.bitwise_not(grayscale_img)

                  # Overwrite the original image with the converted and inverted image
                  cv2.imwrite(image_path, inverted_grayscale_img)

  
    duration = datetime.now() - start
    print("Time taken for conversion and inversion of images: ", duration)


# Main
    
# Set the directory path containing the images
directory = "ascii_dataset"

# Function calls
convert_invert(directory)
