from keras.models import load_model
from keras_face.library.face_net import FaceNet
import pickle
import cv2
import os

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc

model_dir_path = './models/fnet_w.h5'
image_dir_path = "./images"

names = os.listdir(image_dir_path)

fnet = FaceNet()
fnet.load_model(model_dir_path)

database = load_obj('database')

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Frontal faces pattern
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
check_dir = "./check/1.jpg"
while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        # Save the captured image into the datasets folder
        img = gray[y:y+h,x:x+w]
        img = cv2.resize(gray[y:y+h,x:x+w], (96, 96))
        cv2.imwrite(check_dir, img)
        dist, identity = fnet.who_is_it(check_dir, database)
        if identity is None:
            print('Individual is not found in database')
    k = cv2.waitKey(200) & 0xff 
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()