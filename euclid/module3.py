import model
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from model import create_model
import lib

def image_to_embedding(image, model):
    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def recognize_face(face_image, input_embeddings, model):

    embedding = image_to_embedding(face_image, model)
    
    minimum_distance = 200
    name = None
    distances = []
    # Loop over  names and encodings.
    for (input_name, input_embeddings) in input_embeddings.items():
        
        for input_embedding in input_embeddings:
            distances.append(np.linalg.norm(embedding-input_embedding))
        
        euclidean_distance = np.amin(distances)
        #print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))

        
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
    
    if minimum_distance < 0.56:
        return str(name)
    else:
        return None

import glob

def create_input_image_embeddings(model):
    input_embeddings = {}
    for dirname in glob.glob("images/*"):
        embs = []
        person_name = os.path.splitext(os.path.basename(dirname))[0]
        for file in glob.glob(str(dirname) + '/*'):
            image_file = cv2.imread(file, 1)
            embs.append(image_to_embedding(image_file, model))
        input_embeddings[person_name] = embs
    return input_embeddings

def recognize_faces_in_cam(input_embeddings, model):
    

    #cv2.namedWindow("Face Recognizer")
    vc = cv2.VideoCapture(0)
   

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        face_image = lib.align_image(img)
        if face_image is not None:
            #face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]   
            identity = recognize_face(face_image, input_embeddings, model)
            

            if identity is not None:
                #img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,255,255),2)
                #cv2.putText(img, str(identity), (4,30), font, 1, (255,255,255), 2)
                print(str(identity))
            else:
                print("Não reconhecido")
        
            key = cv2.waitKey(100)
            #cv2.imshow("Face Recognizer", img)

            if key == 27: # exit on ESC
                break
        else:
            print("Nenhuma face detectada!")
    vc.release()
    cv2.destroyAllWindows()

model = create_model()
model.load_weights('nn4.small2.v1.h5')

input_embeddings = create_input_image_embeddings(model)
recognize_faces_in_cam(input_embeddings, model)
