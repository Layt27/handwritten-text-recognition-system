# Imports
import cv2
import os
import matplotlib.pyplot as plt
import easyocr

# -------------------------------------------------------------------------------------------------------------

def easy_ocr_detection():

    # Load image using OpenCV
    img_path = "test_imgs/test_img_1.jpg"       # Replace with path to image you want to perform detection and recognition on
    img = cv2.imread(img_path)

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    try:
        # Use EasyOCR to perform OCR and get the coordinates of the bounding boxes
        results = reader.readtext(img)
    except Exception as e:
        print(f"Error during OCR: {e}")
        return

    # Extract coordinates of the detected text regions
    text_coordinates = [result[0] for result in results]

    # Add extra space to the cropping region
    padding = 120
    x_min = max(0, int(min(coord[0][0] for coord in text_coordinates)) - padding)
    y_min = max(0, int(min(coord[0][1] for coord in text_coordinates)) - padding)
    x_max = min(img.shape[1], int(max(coord[2][0] for coord in text_coordinates)) + padding)
    y_max = min(img.shape[0], int(max(coord[2][1] for coord in text_coordinates)) + padding)

    # Crop the image based on the bounding box
    cropped_img = img[y_min:y_max, x_min:x_max]

    # Create a folder to save the bounding box images
    os.makedirs("bbox_imgs", exist_ok=True)

    # Declare and initialize 
    count = 0

    # Draw bounding boxes on the image
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        # Adjust the coordinates to match the cropped region
        adjusted_top_left = (max(0, top_left[0] - x_min), max(0, top_left[1] - y_min))
        adjusted_bottom_right = (min(x_max - x_min, bottom_right[0] - x_min), min(y_max - y_min, bottom_right[1] - y_min))
        # Draw the rectangle on the cropped image
        cv2.rectangle(cropped_img, adjusted_top_left, adjusted_bottom_right, (0, 255, 0), 2)

        # Extract each bounding box region from the cropped image
        each_bbox = cropped_img[adjusted_top_left[1]:adjusted_bottom_right[1], adjusted_top_left[0]:adjusted_bottom_right[0]]

        # Save the bounding box region as an image
        count += 1
        cv2.imwrite(f"bbox_imgs/bbox_{count}.jpg", each_bbox)

    # Display the cropped image
    plt.imshow(cropped_img)
    plt.title("Detected Text")
    plt.show()

# -------------------------------------------------------------------------------------------------------------

# Main
        
# Function call
easy_ocr_detection()