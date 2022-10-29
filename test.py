import cv2
from cv2 import threshold
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

results={
    0:'Bird',
    1:'Drone',
    2:'Helicopter',
    3:'Jet',
    4:'Airbus'
}

path = 'dataset\cascade.xml'
cameraNo = 1                     
objectName = 'Object'       
frameWidth= 1280                    
frameHeight = 720                 
color= (255,0,255)

threshold = 0.45
nms_threshold = 0.2

model = load_model('dataset\model.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cascade = cv2.CascadeClassifier(path)

while True:
    cap.set(10, 130)
    success, img = cap.read()
    img = cv2.resize(img, (150, 150)) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaleVal = 300
    neig = 10
    objects = cascade.detectMultiScale(gray,scaleVal, neig)

    for (x,y,w,h) in objects:
        area = w*h
        minArea = 1
        if area >minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            roi_color = img[y:y+h, x:x+w]

            img_array = np.array(img).astype('float32')

            img_array = np.expand_dims(img_array, axis=0)
            classes = model.predict(img_array,batch_size=32)

            count = 1
            for x in classes:
                for y in x:
                    count = count + 1
                    if y == 1:
                        if count <= 4:
                            print(count)
                            print(results[count])
    
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break