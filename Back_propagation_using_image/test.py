import os
import numpy as np
import cv2
import pickle

def recognise():
    print("starting recogniser...")
    path = os.getcwd()
    path = os.path.join(path, "dataset")
    Categories = ["Modi","Joe"]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer1 = cv2.face.FisherFaceRecognizer_create()
    pickle_in = open("training_img.pickle", "rb")
    x = pickle.load(pickle_in)
    pickle_in = open("training_roll_no.pickle", "rb")
    y = pickle.load(pickle_in)
    recognizer.read("trainer.yml")
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    frame = cv2.imread("test.jpg",1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (150, 150))
        y1, conf = recognizer.predict(roi_gray)
        print(Categories[y1])
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            frame, Categories[y1], (x, y), font, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.imshow('img', frame)
    cv2.waitKey(0)

recognise()