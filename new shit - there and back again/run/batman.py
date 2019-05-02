import numpy as np
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

while True:
	ret, frame = cap.read()
	if ret == True:
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame', frame)
		if cv2.waitKey(30) & 0xFF == ord('q'):
			break