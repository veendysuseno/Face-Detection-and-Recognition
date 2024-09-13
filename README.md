# Face Detection and Recognition

This project captures video from a camera, detects faces in the video stream, recognizes faces using a trained model, and saves the results to a video file and a CSV file.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in the video stream.
- **Face Recognition**: Utilizes an LBPH (Local Binary Patterns Histograms) face recognizer to identify known faces.
- **Video Recording**: Records the video with detected faces annotated with names or labels.
- **CSV Logging**: Logs recognized faces and timestamps to a CSV file.

## Prerequisites

- Python 3.12.5 (e.g. v3.x)
- OpenCV
- NumPy
- PIL (Python Imaging Library)

## requiments.txt

- numpy==1.26.4
- opencv-python==4.10.0.84
- pillow==10.4.0

## Installation

1. **Clone the repository:**

```sh
git clone https://github.com/veendysuseno/Face-Detection-and-Recognition
cd face-detection-recognition
```

2. Install the required Python packages:

```bash
pip install numpy opencv-python pillow
```

2. or install requirements.txt

```bash
pip install -r requirements.txt
```

3. Download Haar Cascade XML File:
   Download haarcascade_frontalface_default.xml from OpenCV GitHub and place it in the haarcascade/ directory.

4. Prepare the Recognizer Training Data:
   Ensure you have trained your LBPH face recognizer and saved the model as training_data.yml in the recognizer/ directory.

## Usage

### Training Data Preparation

If you haven't trained your recognizer yet, use the train.py script to prepare the training data. (You can create a separate script for training if needed.)

### Running the Detection

To start face detection and recognition, run:

```sh
python detection_excel.py -o <path_to_output_csv>
```

### Example:

```sh
python detection_excel.py -o recognized_faces.csv
```

## Notes

- Ensure that the haarcascade_frontalface_default.xml file is correctly placed in the haarcascade/ directory.
- Make sure the training_data.yml file is correctly placed in the recognizer/ directory.
- The output video will be saved in the video/ directory, and filenames will be unique to avoid overwriting.

## Troubleshooting

- Error: Unable to read from camera.
  - Ensure your camera is properly connected and accessible.
- Error: Can't open file 'haarcascade_frontalface_default.xml'
  - Verify that the Haar Cascade XML file is in the correct directory (haarcascade/).

#### @Copyright 2020 | Veenbotronik
