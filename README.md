# Sign-language-recognition-system-using-deep-learning

Table of Contents
Project Overview
Features
Installation
Usage
Model Architecture
Dataset
Results
Web Application (Flask)
Contributing
License
Acknowledgments
Project Overview
This project implements a Sign Language Recognition System using deep learning techniques. It recognizes hand gestures from live video or images, translating them into corresponding letters of the alphabet using a Convolutional Neural Network (CNN). The system is integrated into a web application using Flask for real-time gesture recognition through a webcam.

Features
Real-time Sign Language Recognition: Recognizes hand gestures through webcam input.
Web Application: Built using Flask to provide a user-friendly interface for sign recognition.
Pretrained Model: Uses a trained CNN model to classify gestures into letters A-Z.
Dynamic Cropping: Hand tracking and dynamic ROI cropping using OpenCV and cvzone's HandTrackingModule.
Data Augmentation: Enhanced training with augmented datasets.
Installation
Prerequisites
Ensure you have the following installed:

Python 3.8+
Virtual environment (optional but recommended)
Required libraries (see below)
Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/your-username/Sign-language-recognition-system-using-deep-learning.git
cd Sign-language-recognition-system-using-deep-learning
Step 2: Create and Activate Virtual Environment (Optional)
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
bash
Copy code
pip install -r requirements.txt
The requirements.txt includes necessary libraries such as:

Flask
TensorFlow / Keras
OpenCV
Numpy
cvzone
Step 4: Download the Pretrained Model
Download the trained model (latest_model.h5) from this link and place it in the project root directory.

Usage
Running the Web Application
Start the Flask server using the following command:
bash
Copy code
python app.py
Open your browser and navigate to http://127.0.0.1:5000/. You’ll see the web interface for real-time sign language recognition.

Select the Sign page to start the webcam and begin recognizing hand gestures in real time.

Example Routes
/: Main landing page for the application.
/sign: Opens the sign language recognition page.
/video_feed: Streams video from the webcam to the web app for real-time recognition.
Options and Flags
--model: Specify a different trained model if required.
--video: Use a pre-recorded video file for recognition instead of webcam input.
Model Architecture
The system uses a CNN-based model for classifying images of hand gestures. Below is an overview of the model's architecture:

Input Layer: Accepts 224x224x3 images.
Convolutional Layers: Extract features from the hand gestures using Conv2D layers.
MaxPooling: Reduces the spatial dimensions of the feature maps.
Batch Normalization & Dropout: Improves the model’s robustness and prevents overfitting.
Fully Connected Layers: For classification into one of 26 classes (A-Z).
Output Layer: Softmax activation for multi-class classification.
The model was trained using the Adam optimizer, categorical cross-entropy loss, and with early stopping and learning rate reduction to improve performance.

Dataset
The system is trained on a custom dataset of hand gestures representing the letters A-Z.

Training Data: Augmented with techniques like zooming, shearing, and horizontal flipping to increase dataset diversity.
Validation and Test Data: Used for evaluating model performance after each epoch.
Dataset Folders:
NewData/TrainData: Contains training images.
NewData/TestData: Contains testing images.
NewData/ValidationData: Contains validation images.
You can use publicly available datasets, such as:

ASL Alphabet Dataset
Results
After training for 10 epochs, the model achieves a test accuracy of 95% on unseen sign language gestures.

Confusion Matrix
Below is an example of the confusion matrix showing the classification performance for each letter:


Accuracy and Loss Graphs
The training and validation accuracy/loss over time:

Training Accuracy: Gradually increases and stabilizes around 95%.
Validation Accuracy: Follows the same trend, confirming that the model generalizes well.

Web Application (Flask)
The web application uses Flask to provide an interface where users can perform real-time sign language recognition.

Routes
/sign: Streams live video from the webcam and displays recognized letters.
/about: Describes the project’s purpose.
/home: Main homepage with project info.
Video Feed
The video feed is captured using OpenCV and processed frame-by-frame to detect hands and recognize gestures.

Templates
The web application includes the following HTML templates located in the templates folder:

index.html: Landing page.
sign.html: Real-time sign recognition page.
about.html: About the project.
