import cv2
import os
from keras_face.library.face_net import FaceNet
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

model_dir_path = './models/fnet_w.h5'
image_dir_path = "./images"

fnet = FaceNet()
fnet.load_model(model_dir_path)

database = load_obj('database')
# Frontal faces pattern
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# For each person, enter one numeric face id
face_id = str(input("Nome da pessoa a ser cadastrada: "))
newdir = os.path.join("Images/", face_id)

try:
    os.mkdir(newdir)
except:
    pass

print("\nStart Capture.\nLook at the camera.\n")

count = 0

while(count<1):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     

        # Save the captured image into the datasets folder
        cv2.imshow('image', img)
        ok = int(input("Confirme se a imagem está ok!"))
        if ok:
            count += 1
            img = cv2.resize(gray[y:y+h,x:x+w], (96, 96))
            cv2.imwrite(newdir + "/" + str(count) + ".jpg", img)
            if face_id in database.keys():
                database[face_id].append(fnet.img_to_encoding(newdir + "/" + str(count) + ".jpg"))
            else:
                database[face_id] = [fnet.img_to_encoding(newdir + "/" + str(count) + ".jpg")]
        else:
            pass

    # Press 'ESC' for exiting video
    k = cv2.waitKey(200) & 0xff 
    if k == 27:
        break


save_obj(database, 'database')

cam.release()
cv2.destroyAllWindows()