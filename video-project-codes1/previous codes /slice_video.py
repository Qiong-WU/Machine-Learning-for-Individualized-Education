import numpy as np
import cv2

filename = "D:\\video\\1.mp4"
#filename = filename[1:]
cap = cv2.VideoCapture(filename)
i = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imwrite("D:\\video\\frames\\" +str(i)+ ".bmp" ,frame)
    i += 1

cap.release()
cv2.destroyAllWindows()