from keras.models import load_model
from keras_face.library.face_net import FaceNet
import pickle
import cv2
import os
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import time

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
start = time.time()
model_dir_path = './models/fnet_w.h5'
image_dir_path = "./images"

names = os.listdir(image_dir_path)

fnet = FaceNet()
fnet.load_model(model_dir_path)

database = load_obj('database')

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=96)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Frontal faces pattern

check_dir = "./check"
try:
    os.mkdir(check_dir)
except:
    pass

end = time.time()
print(end-start)
while(True):

    ret, img = cam.read()
    if img is not None:
        img = imutils.resize(img, width=96)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 2)
        if len(faces):
            for face in faces:

                img = fa.align(img, gray, face)
                cv2.imwrite(os.path.join(check_dir,'1.jpg'), img)
                dist, identity = fnet.who_is_it(os.path.join(check_dir,'1.jpg'), database)
                if identity is None:
                    print('Individual is not found in database')
    k = cv2.waitKey(200) & 0xff 
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()