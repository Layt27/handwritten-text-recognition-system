# Imports
import cv2
import matplotlib.pyplot as plt
import easyocr

# -------------------------------------------------------------------------------------------------------------

def easy_ocr_detection():
    # Code for showing bounding boxes on letters using EasyOCR
    
    # Load image using OpenCV
    img_path = "test_imgs/test_img_10_rgb.jpg"
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


    count = 0
    # Extract each bounding box based on the detected bounding boxes
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x, y, w, h = int(top_left[0]), int(top_left[1]), int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
        each_bbox = img[y:y + h, x:x + w]

        # Display each bounding box
        plt.imshow(each_bbox)
        plt.title("Detected Characters")
        plt.show()

        # # Save each bounding box to a folder
        # count +=1
        # cv2.imwrite(f"bbox_imgs/bbox_{count}.jpg", each_bbox)

# -------------------------------------------------------------------------------------------------------------

# Main
        
# Function calls
easy_ocr_detection()