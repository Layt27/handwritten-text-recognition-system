# Handwritten Text Recognition System

This repository contains the files necessary to carry out text recognition.

## Prerequisites

* You need to have Python 3.11.3 installed on your computer. You can download that specific version of Python from the official website.
* You need to have Tesseract-OCR installed on your computer. Follow the steps in section 'Setting up Tesseract-OCR' for directions on installing.

### Setting up Tesseract-OCR

1. Install the Tesseract-OCR from https://github.com/UB-Mannheim/tesseract/wiki.
2. Create a new user variable named "TESSDATA_PREFIX" and add the path of the "tessdata" folder as the value.

## Installation

* Run the requirements.txt file to install the dependencies.
* Use the following command: 'pip install -r requirements.txt'.

## Training the model
 
* Run the 'densenet_train.py' and 'vgg19_train.py' files to train the models.
* The models will be saved in a folder named 'models'.

## Running the system

1. Replace the value of the img_path variable in the 'easyocr_detection.py' file with the path to the image you want to perform detection and recognition on.
2. Run the 'easyocr_detection.py' file.
3. Run the 'pytesseract_detection.py' file.
4. Run the 'text_recognition.py' file to start text recognition.

## Note

* The dataset used for this system was obtained from https://github.com/sueiras/handwritting_characters_database.
* The version of Tesseract-OCR used in this system was tesseract-ocr-w64-setup-5.3.3.20231005.
* The 'used_functions.py' file contains functions that were used to contribute towards the completed system.
* This system was developed to operate on Windows OS.
