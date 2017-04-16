import numpy as np
import cv2

videoPath = "G:\\vs2010\\59NetWork\\video\\IMG_2263.MOV"
cascPath = "G:\\vs2010\\59NetWork\\haarcascade_frontalface_alt2.xml"
eye_cascPath = "G:\\vs2010\\59NetWork\\haarcascade_eye.xml"
outPath = "G:\\vs2010\\59NetWork\\video\\test_result2263.avi"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
#gray = cv2.imread(outPath)

#Open the Camera
cap = cv2.VideoCapture(videoPath)#
out = cv2.VideoWriter(outPath,-1,29.0,(640,480),False)
counter = 0
print cap.get(3)
#for suppress the unstable rectangle
dist_threshold = 20
centers = [[0,0,1,1]]  #initialize to (0,0) and 1 hit 1 counter 
eye_hit = [[True,0]] #[if eyes are detected in previous frame, accumulate miss hit]

def find_nearest_center(center):
    nearest_index = -1
    min_dist = 999999
    b = center[:2]
    for index,item in enumerate(centers):
        a = centers[index][:2]
        dist = np.linalg.norm(np.asarray(a)-np.asarray(b))
        if dist<dist_threshold:
            min_dist = dist
            nearest_index = index
    return nearest_index

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
       # Read the image
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       gray = cv2.resize(gray,(640,480))
       # Detect faces in the image
       faces = faceCascade.detectMultiScale(
             gray,
             scaleFactor=1.1,
             minNeighbors=5,
             minSize=(5,5),
             maxSize=(50,50),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
       )
       for ind,t in enumerate(centers):
           t[3] += 1
           if t[3] >= 30 and t[2] < t[3]*0.5:
               del eye_hit[ind]
               del centers[ind]

       #print("Found {0} faces!".format(len(faces)))
       # Draw a rectangle around the faces
       for (x, y, w, h) in faces:
            #roi_gray = cv2.resize(roi_gray,None,fx=0.5,fy=0.5)
            center = [x+w/2,y+h/2,1,1] #center=[axis_x,axis_y,hit,counter]
            index = find_nearest_center(center)
            if index != -1:
                centers[index][2] += 1 #hit
                #update centers[index]
                centers[index][0] = (centers[index][0]+center[0])/2.0
                centers[index][1] = (centers[index][1]+center[1])/2.0
                if centers[index][2] >= 30:
                   cv2.rectangle(gray, (x, y), (x+w, y+h), 0, 2)
                   roi_gray = gray[y:y+h,x:x+w]
                   roi_gray = cv2.resize(roi_gray,(640,480),interpolation=cv2.INTER_AREA)
                   eyes = eyeCascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=1,
                        minSize=(1,1),
                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                    )
                   
                   detected = False
                   for (ex,ey,ew,eh) in eyes: # eye is supposed to be in the upper half area
                      if ey+eh/2 < 320:
                        eye_hit[index][0] = True
                        eye_hit[index][1] = 0
                        detected = True
                        break
                   if not detected :
                       eye_hit[index][0] = False
                       eye_hit[index][1] += 1
                   if eye_hit[index][1] >= 20:
                       cv2.putText(gray,"sleeping!",(x+w,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,55)

                   h_ratio = float(h)/640
                   w_ratio = float(w)/480
                   roi_gray = cv2.resize(roi_gray,(w,h))


                   for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(gray,(x+int(ex*w_ratio),y+int(ey*h_ratio)),(x+int((ex+ew)*w_ratio),y+int((ey+eh)*h_ratio)),255,1)


                   #cv2.imshow("Faces found", roi_gray)
                   #cv2.waitKey(1)

            
            else:
                centers.append(center)
                eye_hit.append([True,0])

       #cv2.imshow("Faces found", gray)
       if cv2.waitKey(1) & 0xFF == ord('q'):
                     break


       out.write(gray)
       counter += 1
       print counter
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()