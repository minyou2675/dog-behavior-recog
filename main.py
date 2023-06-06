from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
# from dotenv import load_dotenv
# load_dotenv()
import os
import pymysql

connection = pymysql.connect(
  host= "aws.connect.psdb.cloud",
  user="s2d94pl1hgjbie54te9f",
  passwd="main-2023-apr-16-irss2f" ,
  db= "yumin-db",
  autocommit = True,
  ssl_mode = "VERIFY_IDENTITY",
  ssl      = {
    "ca": "/etc/ssl/cert.pem"
  }
)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./model/keras_model.h5", compile=False)

# Load the labels
class_names = open("./model/labels.txt", "r",encoding='utf-8').readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture('./video2.mp4')

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    if np.round(confidence_score * 100) > 90:
         print("Class:", class_name[2:], end="")
         print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")


    # Print prediction and confidence score
   

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
