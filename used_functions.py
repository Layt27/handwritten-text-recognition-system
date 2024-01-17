# Imports
import os
import cv2
from datetime import datetime

# -------------------------------------------------------------------------------------------------------------------

# Functions

# Invert colors of images function
def invert_colors(directory):
    start = datetime.now()
    print("Inverting colors of images...")

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

                # Check if the file is an image file
                if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                  # Read the image using OpenCV
                  img = cv2.imread(image_path)

                  # Invert the colors of the image (reverse intensity values)
                  inverted_img = cv2.bitwise_not(img)

                  # Overwrite the original image with the inverted image
                  cv2.imwrite(image_path, inverted_img)

  
    duration = datetime.now() - start
    print("Time taken for inverting the colors of images: ", duration)


# Get image shape of images function
def print_image_shape(directory):
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

                # Check if the file is an image file
                if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                  # Read the image using OpenCV
                  img = cv2.imread(image_path)

                  # Get image shape with number of channels
                  image_shape = img.shape       # This will return a tuple (height, width, channels)
                  print(f"Folder name: {folder}, File name: {file}, Shape: {image_shape}")
                  return


# Main
    
# Set the directory path containing the images
directory = "ascii_dataset"

# Function calls
invert_colors(directory)
print_image_shape(directory)