import numpy as np
import cv2
import os
import time
import datetime
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="barcodes.csv",
                help="path to output CSV file containing barcodes")
args = vars(ap.parse_args())

# Initialize camera
cam = cv2.VideoCapture(0)

# Load face detection model
faceDetect = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Load trained face recognizer model
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/training_data.yml")

# Create a directory for videos if it doesn't exist
if not os.path.exists("video"):
    os.makedirs("video")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
i = 0  # Video file index
video_filename = f'video/output{i}.avi'

# Check if video file already exists and increment index
while os.path.exists(video_filename):
    i += 1
    video_filename = f'video/output{i}.avi'

out = cv2.VideoWriter(video_filename, fourcc, 10.0, (640, 480))

# Open CSV file for writing recognized IDs
csv = open(args["output"], "w")
found = set()  # Set to keep track of recognized faces

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Unable to read from camera.")
        break
    
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Predict ID using the recognizer
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        
        # Assign a label to the detected ID
        if id == 1:
            text = "VEENDY"
        else:
            text = "UNKNOWN"
        
        # Write recognized face to CSV file if not already found
        if text not in found:
            csv.write("{},{}\n".format(datetime.datetime.now(), text))
            csv.flush()
            found.add(text)
        
        # Display the label on the video feed
        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        
    # Write frame to video output
    out.write(img)
    
    # Display the frame with face detection
    cv2.imshow('frame', img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
out.release()
csv.close()
cv2.destroyAllWindows()
