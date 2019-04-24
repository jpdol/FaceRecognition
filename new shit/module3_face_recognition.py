from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import lib

classifier = load_model('classifier.h5')
facenet = load_model('facenet_keras.h5')

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['Jean', 'Lorena', 'Diego', 'Hermann', 'Professora'] 

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Frontal faces pattern
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        # Save the captured image into the datasets folder
        img = gray[y:y+h,x:x+w]
	aux = lib.prewhiten(img)
	emb = facenet.predict(aux)
	emb = l2_normalize(np.concatenate([emb]))
	
        y_pred = classifier.kneighbors(emb)[0]

        for i in range(len(y_pred)):
            for j in range(len(y_pred[i])):
                if (y_pred[i][j]==min(y_pred[i])):
                    print(names[j], '-', max(y_pred[i]))
        

cam.release()
cv2.destroyAllWindows()