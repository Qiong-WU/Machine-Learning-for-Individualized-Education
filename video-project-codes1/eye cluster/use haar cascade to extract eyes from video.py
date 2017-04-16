import numpy as np
from sklearn import mixture
from sklearn.ensemble import RandomForestRegressor
import cv2
import csv


# Get user supplied values
videoPath = "G:\\vs2010\\59NetWork\\video\\IMG_2263.MOV"
cascPath =  "G:\\vs2010\\59NetWork\\frontalFace10\\haarcascade_frontalface_alt2.xml"#"G:\\vs2010\\59NetWork\\haarcascade_frontalface_alt2.xml" # # #"G:\\vs2010\\59NetWork\\haarcascade_frontalface_default.xml"
lefteye_cascPath = "G:\\vs2010\\59NetWork\\haarcascade_mcs_lefteye.xml" #"G:\\vs2010\\59NetWork\\haarcascade_eye.xml"
righteye_cacsPath = "G:\\vs2010\\59NetWork\\haarcascade_mcs_righteye.xml"
mouth_cascPath = "G:\\vs2010\\59NetWork\\Mouth.xml" #"G:\\vs2010\\59NetWork\\haarcascade_mcs_mouth.xml"
outPath = "G:\\vs2010\\59NetWork\\video\\test_result2269_mouth_detect.avi"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
lefteyeCascade = cv2.CascadeClassifier(lefteye_cascPath)
righteyeCascade = cv2.CascadeClassifier(righteye_cacsPath)
mouthCascade = cv2.CascadeClassifier(mouth_cascPath)
#gray = cv2.imread(outPath)

#Open the Camera
cap = cv2.VideoCapture(videoPath)#
#cap = cv2.VideoCapture(0);
#out = cv2.VideoWriter(outPath,-1,29.0,(640,480),False)
counter = 0
print cap.get(3)
#for suppress the unstable rectangle
dist_threshold = 20
centers = [[0,0,1,1]]  #initialize to (0,0) and 1 hit 1 counter 
eye_hit = [[True,0]] #[if eyes are detected in previous frame, accumulate miss hit]
mean_value = [[-1,-1,0]]  #store the mean value of lower half face to assert if speaking
                      #[mean_value[t-2], mean_value[t-1], miss hit]
alpha = 0.2 
beta = 0.4 #using for calcualte the moving average, which is defined as mean_value(t+1) = alpha*mean_value(t-2)+beta*mean_value(t-1)+(1-alpha-beta)*new_mean
mean_value_recorder = [[]]

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

eye_location = []
record_flag = False;
get_count = 0
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
       #if get_count < 20:
       #    cv2.imwrite("G:\\vs2010\\59NetWork\\"+str(get_count)+".png",frame)
       #    get_count += 1

       # Read the image
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       gray = cv2.resize(gray,(640,480))
       # Detect faces in the image
       faces = faceCascade.detectMultiScale(
             gray,
             scaleFactor=1.1,
             minNeighbors=5,
             minSize=(1,1),
             maxSize=(50,50),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
       )

       #t[3] is the frame counter, t[2] is the eye-hit counter, when eye missed more than
       #50% among the past frames and 30 frames have passed, the center will be deleted, 
       for ind,t in enumerate(centers):
           t[3] += 1
           if t[3] >= 30 and t[2] < t[3]*0.5:
               del eye_hit[ind]
               del centers[ind]

       # Draw a rectangle around the faces
       face_counter = -1
       for (x, y, w, h) in faces:
            face_counter += 1
            #print x,y,w,h
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
                   #cv2.imshow("eyes region", roi_gray)
                   lefteyes = lefteyeCascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(1,1),
                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                    )
                   righteyes = righteyeCascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(1,1),
                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                    )
                   detected = False
                   if isinstance(lefteyes,tuple):
                       if isinstance(righteyes,tuple):
                           eyes = np.empty(shape=(1,4))
                       else:
                           righteye = np.ndarray(shape=(1,4))
                           righteye[0] = righteyes[0]
                           eyes = righteye
                   else:
                       if isinstance(righteyes,tuple):
                           lefteye = np.ndarray(shape=(1,4))
                           lefteye[0] = lefteyes[0]
                           eyes = lefteye
                       else:
                           righteye = np.ndarray(shape=(1,4))
                           righteye[0] = righteyes[0]
                           lefteye = np.ndarray(shape=(1,4))
                           lefteye[0] = lefteyes[0]
                           eyes = np.concatenate((righteye,lefteye))

                   for (ex,ey,ew,eh) in eyes: # eye is supposed to be in the upper half area
                      if ey+eh/2 < 320:
                        eye_hit[index][0] = True
                        eye_hit[index][1] = 0
                        detected = True
                        break
                   if not detected :
                       eye_hit[index][0] = False
                       eye_hit[index][1] += 1
                   #if eye_hit[index][1] >= 20:
                       #cv2.putText(gray,"sleeping!",(x+w,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,55)

                   h_ratio = float(h)/640
                   w_ratio = float(w)/480
                    
                   for (ex,ey,ew,eh) in eyes:
                       if ey+eh/2 < 320 and face_counter == 1:
                           eye_location.append([x+int(ex*w_ratio)-5,y+int(ey*h_ratio)+5,x+int((ex+ew)*w_ratio)-5,y+int((ey+eh)*h_ratio)+5])
                           #cv2.rectangle(gray,(x+int(ex*w_ratio)-5,y+int(ey*h_ratio)+5),(x+int((ex+ew)*w_ratio)-5,y+int((ey+eh)*h_ratio)+5),255,1)
                        #if  x+int(ex*w_ratio)>320:
                            #print x+int(ex*w_ratio),y+int(ey*h_ratio),x+int((ex+ew)*w_ratio),y+int((ey+eh)*h_ratio)

                    
                   ##begin mouth detection, mouth is supposed to be the lower half of the face
                   #roi_gray = gray[int(y+h/4.0*3.0):y+h,int(x):x+w]
                   #roi_gray = cv2.resize(roi_gray,(240,160),interpolation=cv2.INTER_AREA)
                   #tt = np.diff(roi_gray,0)
                   #tt = tt[:,:-1]
                   #calculate_object = np.sum(tt + np.diff(roi_gray,1))/np.prod(np.shape(tt))
                   #if (mean_value[index][0] == -1 ):
                   #    mean_value[index][0] = calculate_object
                   #    mean_value_recorder[index].append(mean_value[index][0])
                   #    continue
                   #elif (mean_value[index][1] == -1):
                   #    mean_value[index][1] = calculate_object
                   #    continue

                   #if ( np.abs(mean_value[index][1]-calculate_object) > 2.1):
                   #    mean_value[index][2] += 1
                   #else:
                   #    if mean_value[index][2] > 0:
                   #        mean_value[index][2] -= 0.15
                   #    else:
                   #        mean_value[index][2] = 0

                   #print mean_value[index][2]
                   #temp = mean_value[index][1]
                   #mean_value[index][1] = alpha*mean_value[index][0] + beta*mean_value[index][1] + (1-alpha-beta)*np.mean(roi_gray)
                   #mean_value[index][0] = temp
                   #mean_value_recorder[index].append(np.mean(roi_gray)) #mean_value[index][1]

                   #if mean_value[index][2] > 2:
                   #    cv2.putText(gray,"speaking!",(x+w,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,55)

                   #cv2.putText(gray,"testing!",(320,240), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,55)

                   #mouths = mouthCascade.detectMultiScale(
                   #     roi_gray,
                   #     scaleFactor=1.5,
                   #     minNeighbors=1,
                   #     minSize=(1,1),
                   #     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                   # )

                   #h_ratio = float(h)/160*4.0/3.0
                   #w_ratio = float(w)/240
                   #for (mx, my, mw, mh) in mouths:
                   #    cv2.rectangle(roi_gray, (mx,my), (mx+mw,my+mh),255,1)
                       #mx = int(mx*w_ratio)
                       #my = int(my*h_ratio)
                       #mw = int(mw*w_ratio)
                       #mh = int(mh*h_ratio)
                       #cv2.rectangle(gray, (x+mx,int(y+my+h/2.0)),(x+mx+mw,int(y+h/2.0+my+mh)),255,1)
                   
                   #cv2.imshow("Faces found", roi_gray)
            else:
                centers.append(center)
                eye_hit.append([True,0])
                #mean_value.append([-1,-1,0])
                #mean_value_recorder.append([])

       #cv2.imshow("Faces found", gray)
       #get_count += 1
       #cv2.imwrite("G:\\vs2010\\59NetWork\\face_data\\"+str(get_count)+".png",gray)

       if cv2.waitKey(1) & 0xFF == ord('q'):
                     break

       #out.write(gray)
       counter += 1
       #print counter
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    else:
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
np.save("G:\\vs2010\\59NetWork\\eye_coordinate\\eye_location.npy",eye_location)