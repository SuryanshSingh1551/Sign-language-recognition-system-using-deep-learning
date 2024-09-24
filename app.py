from flask import redirect
from flask import Flask, render_template, Response, redirect, url_for
from flask import Flask, render_template, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__, static_folder='static')

# Load the trained model
model = load_model("latest_model.h5")

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Define class labels
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Function to capture video from the webcam and perform sign language recognition
# Function to capture video from the webcam and perform sign language recognition
def get_video():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()

        # Find hands in the frame
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            offset = int(0.2 * max(w, h))

            # Crop and resize hand region of interest
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Check if imgCrop is not empty
            if imgCrop.size != 0:
                imgResize = cv2.resize(imgCrop, (224, 224))

                # Preprocess the image for prediction
                imgNormalize = imgResize / 255.0
                imgArray = np.expand_dims(imgNormalize, axis=0)

                # Make predictions
                predictions = model.predict(imgArray)
                class_index = np.argmax(predictions)
                class_label = class_labels[class_index]

                # Display predicted class label
                cv2.putText(img, class_label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert the frame to a byte array
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/redirect_index')
def redirect_index():
    return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    return Response(get_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/redirect_home')
def redirect_home():
    return redirect(url_for('home'))

@app.route('/')
def redirect_to_sign():
    return redirect(url_for('sign'))

@app.route('/sign')
def sign():
    return render_template('sign.html')

@app.route('/about')
def redirect_about():
    return redirect(url_for('about'))

@app.route('/redirect_about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
