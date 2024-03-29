# Imports
import cv2
import os
import pytesseract

# -------------------------------------------------------------------------------------------------------------

"""
Run the command `tesseract --help-oem` in the terminal for below information.

OCR Engine Modes (oem)
0   Legacy engine only.
1   Neural nets LSTM engine only.
2   Legacy + LSTM engines.
3   Default, based on what is available.
"""

"""
Run the command `tesseract --help-psm` in the terminal for below information.

Page Segementation Modes (psm)
0   Orientation and script detection (OSD only).
1   Automatic page segmentation with OSD.
2   Automatic page segmentation w/o OSD or OCR.
3   Fully automatic page segmentation w/o OSD. (Default)
4   Assume a single column of text of variable sizes.
5   Assume a single uniform block of vertically aligned text.
6   Assume a single uniform block of text.
7   Treat the image as a single text line.
8   Treat the image as a single word.
9   Treat the image as a single word in a circle.
10  Treat the image as a single character.
11  Sparse text. Find as much text as possible in no particular order.
12  Sparse text with OSD.
13  Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
"""

# -------------------------------------------------------------------------------------------------------------

def pytess_rec(bbox_directory, char_list):
    # Specify custom configuration for the Tesseract OCR engine
    custom_config = r"--psm 8 --oem 3"

    # Create a folder to save the bounding box images
    os.makedirs("char_imgs", exist_ok=True)

    # Declare and initialize count variable used in naming images and keeping track of how many characters make up each bounding box
    char_count = 0

    # Store names of files from bbox_imgs directory
    files = os.listdir(bbox_directory)
    
    # Loop through each file in the folder
    for file in files:
        # Store the image path
        img_path = os.path.join(bbox_directory, file)

        if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            # Read the image using OpenCV
            img = cv2.imread(img_path)

            # Extract information about the bounding boxes of characters in an image
            boxes = pytesseract.image_to_boxes(img, config = custom_config)
        else:
            print("Invalid image format provided. Ensure the directory contains images of type (.jpg, .jpeg, .png).")

        # Draw bounding boxes on each character
        for box in boxes.splitlines():
            box = box.split(" ")

            # Extract bounding box coordinates
            x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            # Extract the character region from the original image
            character_bbox = img[y1:y2, x1:x2]

            # Save each extracted character region as an image
            char_count += 1
            cv2.imwrite(f"char_imgs/char_{char_count}.jpg", character_bbox)

        # Append the count of the last character in every bounding box to the list 
        char_list.append(char_count)

# -------------------------------------------------------------------------------------------------------------

# Main

bbox_directory = "bbox_imgs"

# Declare and initialize a list for storing the count of the last character that makes up each bounding box
char_list = []

# Function call
pytess_rec(bbox_directory, char_list)