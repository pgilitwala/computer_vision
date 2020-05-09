# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:22:08 2020

@author: pgilitwala
"""

import cv2
faceCd = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCd = cv2.CascadeClassifier("haarcascade_eye.xml")
smileCd = cv2.CascadeClassifier("haarcascade_smile.xml")

def detect(gray, frame):
    faces = faceCd.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCd.detectMultiScale(roi_gray, 1.1, 22)
        smile = smileCd.detectMultiScale(roi_gray, 1.7, 22)
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        for (sx, sy, sw, sh) in smile[:1]:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
        
    return frame

vc = cv2.VideoCapture(0)
while True:
    _, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()