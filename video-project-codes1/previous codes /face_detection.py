import cv2
import sys
imgPath = "G:\\vs2010\\59NetWork\\9.jpg"
cascPath = "G:\\vs2010\\59NetWork\\haarcascade_frontalface_alt2.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imgPath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
   gray,
   scaleFactor=1.1,
   minNeighbors=5,
   minSize=(3,3),
   flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
print("Found {0} faces!".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.imwrite("G:\\vs2010\\59NetWork\\9_face.jpg",image);