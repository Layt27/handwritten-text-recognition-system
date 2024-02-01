# Imports
import cv2
import matplotlib.pyplot as plt
import pytesseract

# -------------------------------------------------------------------------------------------------------------

"""
OCR Engine Modes (oem)
0   Legacy engine only.
1   Neural nets LSTM engine only.
2   Legacy + LSTM engines.
3   Default, based on what is available.
"""

"""
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

def pytess_rec():
    # Load image using OpenCV
    img = cv2.imread("test_imgs/test_img_1.jpg")
    # "bbox_imgs/bbox_1.jpg"
    # "test_imgs/test_img_9.jpg"

    # Extract the image shape
    height, width, _ = img.shape

    # Specify custom configuration for teh Tesseract OCR engine
    custom_config = r"--psm 6 --oem 3"

    # Extract information about the bounding boxes of characters in an image
    boxes = pytesseract.image_to_boxes(img, config = custom_config)
    print(boxes)

    # Draw bounding boxes on each character
    for box in boxes.splitlines():
        box = box.split(" ")
        img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0), 2)
    
    # Display the image with bounding boxes
    plt.imshow(img)
    plt.title("Image with Character Boxes")
    plt.show()

# -------------------------------------------------------------------------------------------------------------

# Main
        
# Function call
pytess_rec()