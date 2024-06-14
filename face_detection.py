import cv2
import numpy as np
import time

face_classifier  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

videoCam = cv2.VideoCapture(0)

if not videoCam.isOpened():
    print("Camera connot be accesed")
    exit()

the_Q_button_is_pressed = False
while (the_Q_button_is_pressed == False):
    scan, framework = videoCam.read()

    if scan == True:
        gray = cv2.cvtColor(framework, cv2.COLOR_BGR2GRAY)
        degree = face_classifier.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 2)

        for (x, y, w, h) in degree:
            cv2.rectangle(framework, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        #print("No of faces detected: ", len(degree))
        text = "No of faces detected = " + str(len(degree))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(framework, text, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Result", framework)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            the_Q_button_is_pressed = True
            break


videoCam.release()
cv2.destroyAllWindows()
