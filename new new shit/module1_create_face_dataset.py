import cv2
import os
import lib


cam = cv2.VideoCapture(0)

count = 0
while(True):
    ret, img = cam.read()
    img = lib.align_image(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    count += 1
    # Save the captured image into the datasets folder
    cv2.imwrite("images/User_" + str(count) + ".jpg", img)
    cv2.imshow('image', img)
    k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 10: # Take 10 face sample and stop video
         break
cam.release()
cv2.destroyAllWindows()
