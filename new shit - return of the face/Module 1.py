import cv2
import os
import face_recognition
import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

cam = cv2.VideoCapture(1)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

model_dir_path = './models/fnet_w.h5'
image_dir_path = "./images"


database = load_obj('database')

print(database)
clr = int(input("Would you like to clear the database?"))
if clr:
    database = dict()
    print("Database Cleared!")
else:
    pass

# Frontal faces pattern
#face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
try:
    os.mkdir(image_dir_path)
except:
    pass
# For each person, enter one numeric face id
face_id = str(input("Nome da pessoa a ser cadastrada: "))
newdir = os.path.join(image_dir_path, face_id)

try:
    os.mkdir(newdir)
except:
    pass


print("\nStart Capture.\nLook at the camera.\n")

count = 0

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while count<4:
	# Grab a single frame of video
	ret, frame = cam.read()
	cv2.imshow('image', frame)
	
	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

	# Only process every other frame of video to save time
	if process_this_frame:
		try:
			ok = int(input("Confirme se a imagem está ok!"))
		except:
			ok = 0
		if ok:
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame)
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
			for face_encoding in face_encodings:
				count += 1
				aux = os.listdir(newdir)
				if len(aux):
					aux = aux[len(aux)-1]
					last = int(aux[:len(aux)-4])
				else:
					last = -1
				cv2.imwrite(newdir + "/" + str(last+1) + ".jpg", frame)
				if face_id in database.keys():
					database[face_id].append(face_encoding)
				else:
					database[face_id] = [face_encoding]
	process_this_frame = not process_this_frame

save_obj(database, 'database')

cam.release()
cv2.destroyAllWindows()