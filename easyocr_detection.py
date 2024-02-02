# Imports
import cv2
import matplotlib.pyplot as plt
import easyocr

# -------------------------------------------------------------------------------------------------------------

def easy_ocr_detection():
    # Code for showing bounding boxes on letters using EasyOCR
    
    # Load image using OpenCV
    img_path = "test_imgs/test_img_6.jpg"
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


    # Draw bounding boxes on the image
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)


    # count = 0
    # # Extract each bounding box based on the detected bounding boxes
    # for (bbox, text, prob) in results:
    #     (top_left, top_right, bottom_right, bottom_left) = bbox
    #     x, y, w, h = int(top_left[0]), int(top_left[1]), int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
    #     each_bbox = img[y:y + h, x:x + w]

    #     # Display each bounding box
    #     plt.imshow(each_bbox)
    #     plt.title("Detected Characters")
    #     plt.show()

        # # Save each bounding box to a folder
        # count +=1
        # cv2.imwrite(f"bbox_imgs/bbox_{count}.jpg", each_bbox)
    

    # Extract coordinates of the detected text regions
    text_coordinates = [result[0] for result in results]

    # Add extra space to the cropping region
    padding = 80
    x_min = max(0, int(min(coord[0][0] for coord in text_coordinates)) - padding)
    y_min = max(0, int(min(coord[0][1] for coord in text_coordinates)) - padding)
    x_max = min(img.shape[1], int(max(coord[2][0] for coord in text_coordinates)) + padding)
    y_max = min(img.shape[0], int(max(coord[2][1] for coord in text_coordinates)) + padding)

    # Crop the image based on the bounding box
    cropped_image = img[y_min:y_max, x_min:x_max]

    # Display the cropped image
    plt.imshow(cropped_image)
    plt.title("Detected Text")
    plt.show()

# -------------------------------------------------------------------------------------------------------------

# Main
        
# Function calls
easy_ocr_detection()