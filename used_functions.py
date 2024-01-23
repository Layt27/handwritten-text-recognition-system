# Imports
import os
import cv2
from datetime import datetime

# -------------------------------------------------------------------------------------------------------------------

# Functions

# Invert colors of images function
def invert_colors(dataset_directory):
    start = datetime.now()
    print("Inverting colors of images...")

    folders = os.listdir(dataset_directory)

    # Loop through each folder in the directory
    for folder in folders:
        folder_path = os.path.join(dataset_directory, folder)

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
def print_image_shape(dataset_directory):
    folders = os.listdir(dataset_directory)

    # Loop through each folder in the directory
    for folder in folders:
        folder_path = os.path.join(dataset_directory, folder)

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


# Convert images to grayscale function
def convert_to_grayscale(test_imgs_directory):
    start = datetime.now()
    print("Converting images to grayscale...")

    files = os.listdir(test_imgs_directory)
    
    # Loop through each file in the folder
    for file in files:
        # Store the test image path
        test_img_path = os.path.join(test_imgs_directory, file)

        if test_img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            # Read the image using OpenCV
            test_img = cv2.imread(test_img_path)

            # Convert the image to grayscale
            grayscale_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(test_img_path, grayscale_test_img)
        else:
            print("Invalid image format provided. Ensure the directory contains images of type (.jpg, .jpeg, .png).")
    
    duration = datetime.now() - start
    print("Time taken for converting images to grayscale: ", duration)
    
# -------------------------------------------------------------------------------------------------------------------

# Main
    
# Set the directory paths containing the images
dataset_directory = "ascii_dataset"
test_imgs_directory = "test_imgs"

# Function calls
invert_colors(dataset_directory)
print_image_shape(dataset_directory)
convert_to_grayscale(test_imgs_directory)