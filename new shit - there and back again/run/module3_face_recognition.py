from keras.models import load_model
from keras_face.library.face_net import FaceNet
import pickle
import cv2
import os
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib

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

ace_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=96)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Frontal faces pattern
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
check_dir = "./check/1.jpg"
while(True):

    ret, img = cam.read()
    img = imutils.resize(img, width=96)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 2)

    for face in faces:

        img = fa.align(img, gray, face)
        cv2.imwrite(check_dir, img)
        dist, identity = fnet.who_is_it(check_dir, database)
        if identity is None:
            print('Individual is not found in database')
    k = cv2.waitKey(200) & 0xff 
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()