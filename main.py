from ultralytics import YOLO
import cv2
import os
import numpy
import pandas
from pathlib import Path
import uuid
import numpy as np
import re
import itertools
import paddleocr
import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
from PIL import Image


print("current directory",os.getcwd())

def license_plate(video_path):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print("Video not loaded successfully")
    else:
        print("Video loaded successfully")

    extracted_frames_directory = r"Extracted_Frames"

    # Check if the directory already exists
    if not os.path.exists(extracted_frames_directory):
        # Create the directory
        os.makedirs(extracted_frames_directory)
        print(f"Directory created: {extracted_frames_directory}")
    else:
        print(f"Directory already exists: {extracted_frames_directory}")

    def getFrame(sec):
        video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # Getting 1 image in every second
        hasFrames, image = video.read()
        if hasFrames:
            cv2.imwrite(f"{extracted_frames_directory}/frame " + str(sec) + " sec.jpg", image)  # Save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 1  # It will capture an image every 1 second
    success = getFrame(sec)
    while success:
        sec = sec + frameRate
        success = getFrame(sec)

    model = YOLO(r"best.pt")
    count=0
    save_dir = r"saved_crop_image"

    # Check if the directory already exists
    if not os.path.exists(save_dir):
        # Create the directory
        os.makedirs(save_dir)
        print(f"Directory created: {save_dir}")
    else:
        print(f"Directory already exists: {save_dir}")

    for i in os.listdir(extracted_frames_directory):
        # Read the current image file using OpenCV
        image = cv2.imread(os.path.join(extracted_frames_directory, i))
        print(os.path.join(extracted_frames_directory, i))

        # Extract the filename and extension of the current image file
        file_name, ext = os.path.splitext(i)

        # Specify the desired filename for the processed image file
        image_file_name = f"{file_name}{count}"

        # Make a prediction on the image
        results = model.predict(source=image, show_labels=False, show_conf=False, verbose=True)

        if len(results[0].boxes.xyxy) == 0:
            print(f"No bounding boxes found in {i}. Skipping...")
            continue

        #annotated_frame = results[0].plot(labels=False)

        # get the coordinates of the first bounding box
        xmin, ymin, xmax, ymax = map(int, results[0].boxes.xyxy[0][:4])

        # crop the bounding box part of the image
        cropped_img = image[ymin:ymax, xmin:xmax]

        # Save the cropped image to the desired directory
        cv2.imwrite(os.path.join(save_dir, f"{image_file_name}.jpg"), cropped_img)

        count += 1
    # Removing the extracted frames    
    for i in os.listdir(extracted_frames_directory):
        os.remove(f'{extracted_frames_directory}/{i}')    

    Paddle = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')

    License_plate_number = []

    for i in os.listdir(save_dir):
        image_path = os.path.join(save_dir,i)
        image = cv2.imread(image_path)

       # Perform OCR on the image
        result = Paddle.ocr(image_path, cls=False, det=True)
        texts = []
        for line in result:
            for box, text in line:
                texts.append(text[0])
                # print('uyfysdf',texts)

        # Remove special characters and perform replace operations on OCR result
        combined_text = re.sub(r'[^a-zA-Z0-9]', '', ''.join(texts)).replace(' ', '').replace('IND', '')
        # print('comd',combined_text)
        # Define a list of valid Indian state codes
        state_codes = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB']

        # Iterate over the state codes
        for state_code in state_codes:
            # Find the index of the state code in the combined_text
            index = combined_text.find(state_code)
            # If the state code is found
            if index != -1:
                # Remove any text before the state code
                combined_text = combined_text[index:]

                break

        
        # Change the 3rd character of combined_text from '0' or 'o' to '0'
        if len(combined_text) >= 3 and (combined_text[2] == 'O' or combined_text[2] == 'o'):
            combined_text = combined_text[:2] + '0' + combined_text[3:]
        if len(combined_text) >= 4 and combined_text[3] == 'I':
            combined_text = combined_text[:3] + '1' + combined_text[4:]

        # Change the 5th character of combined_text from '8' to 'B'
        if len(combined_text) >= 5 and combined_text[4] == '8':
            combined_text = combined_text[:4] + 'B' + combined_text[5:]
            
        # Add the OCR result to the License_plate_number list
        License_plate_number.append(combined_text)
        print('lice',License_plate_number)
    # Removing the cropped images
    for i in os.listdir(save_dir):
        os.remove(f'{save_dir}/{i}')
    # Define a list of valid Indian state codes
    state_codes = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB']

    # Define the regular expression pattern for Indian license plate numbers
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}$'

    # Create a list to store the valid license plate numbers
    valid_license_plate_numbers = []

    # Iterate over the License_plate_number list
    for license_plate_number in License_plate_number:
        # Check if the current license plate number starts with a valid state code
        if license_plate_number[:2] in state_codes:
            # Check if the current license plate number matches the pattern
            if re.match(pattern, license_plate_number):
                # If it matches, add it to the valid_license_plate_numbers list
                valid_license_plate_numbers.append(license_plate_number)
    # Remove duplicates from the valid_license_plate_numbers list
    valid_license_plate_numbers = list(set(valid_license_plate_numbers))
     # Check if any valid license plate numbers are found
    if len(valid_license_plate_numbers) == 0:
        return "The image is not clear"
    Final_license_number = max(valid_license_plate_numbers, key=len)
    return Final_license_number

def set_background_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://www.marshallsindia.com/ImageBuckets/ItemImages/ZA%201903.jpg?id=75');
            background-size: cover;
        }}
        .css-18ni7ap {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 0rem;
            outline: none;
            z-index: 999990;
            display: block;
        }}
        .logo {{
            position: absolute;
            top: 10px;
            right: 10px;
            width: 200px;
            height: 200px;
            background-image: url('https://dn3ipw86p78eg.cloudfront.net/settings/59567410logopng.png');
            background-size: contain;
            background-repeat: no-repeat;
        }}
        </style>
        <div class="logo"></div>
        """,
        unsafe_allow_html=True
    )

def main():
    set_background_image()

    st.markdown("<h1 style='text-align: center; color: blue;'font-size:30px; font-family:Helvetica;'>Automatic Number Plate Detection</h1>", unsafe_allow_html=True)
    col11,  col31, col41 = st.columns([ 3, 5, 3])
    with col11:
        st.write("")
    with col31:
        
        st.markdown(
    """
    <style>
        .uploadedFileName {
            color: green;
        }
    </style>
    """,
    unsafe_allow_html=True
)
    with st.sidebar:
        st.header("Upload  image")
        uploaded_file1 = st.file_uploader("", type=['jpg','jpeg','png'])
    with col41:
        st.write("")
    col1, col2 = st.columns([3,2])
    with col1:
        if uploaded_file1 is not None:
            # Display uploaded video
            image = Image.open(uploaded_file1,)
            st.image(image, caption='Uploaded Photo', width=350)
    with col2:
        if uploaded_file1 is not None:
            image = Image.open(uploaded_file1,)
            st.markdown("<h1 style='text-align: center; color: black;'font-size:05px; font-family:Helvetica;'>Number Plate</h1>", unsafe_allow_html=True)
            with open("photo.jpg", "wb") as f:
                    f.write(uploaded_file1.getbuffer())
                    lic_numbr = license_plate("photo.jpg")
            st.markdown(f"<h1 style='text-align: center; color: black;'font-size:05px; font-family:Helvetica;'>{lic_numbr}</h1>", unsafe_allow_html=True,)
        
        #st.video(uploaded_file1)
        
    #     with open("video.mp4", "wb") as f:
    #         f.write(uploaded_file1.getbuffer())
    #         lic_numbr = license_plate("video.mp4")
    #         st.markdown(f"<h1 style='text-align: center; color: black;'font-size:05px; font-family:Helvetica;'>{lic_numbr}</h1>", unsafe_allow_html=True)
    # # Specify the directory to store the uploaded video
        
    # # Use the stored video path in the license_plate function
    # lic_num = license_plate(video_path)
    # st.write(lic_num)


if __name__ == "__main__":
    main()
