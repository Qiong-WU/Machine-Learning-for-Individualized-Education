import numpy as np
from sklearn import mixture
from sklearn.ensemble import RandomForestRegressor
import cv2
import csv

ProjectFolder = "G:\\vs2010\\59NetWork\\identity_state_project\\"
cascFolder = "cascade\\"
# Get user supplied values
videoPath = ProjectFolder+"video\\IMG_2263.MOV"
cascPath =  ProjectFolder+cascFolder+"haarcascade_frontalface_alt2.xml"#"G:\\vs2010\\59NetWork\\haarcascade_frontalface_alt2.xml" # # #"G:\\vs2010\\59NetWork\\haarcascade_frontalface_default.xml"
lefteye_cascPath = ProjectFolder+cascFolder+"haarcascade_mcs_lefteye.xml" #
righteye_cacsPath = ProjectFolder+cascFolder+"haarcascade_mcs_righteye.xml"
eye_cacsPath = ProjectFolder+cascFolder+"haarcascade_mcs_eyepair_small.xml"
mouth_cascPath = ProjectFolder+cascFolder+"Mouth.xml"
outPath = "G:\\vs2010\\59NetWork\\video\\test_result2269_mouth_detect.avi"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
lefteyeCascade = cv2.CascadeClassifier(lefteye_cascPath)
righteyeCascade = cv2.CascadeClassifier(righteye_cacsPath)
eyeCascade = cv2.CascadeClassifier(eye_cacsPath)
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
centers = []  #initialize to (0,0) and 1 hit 1 counter 
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
               del centers[ind]

       # Draw a rectangle around the faces
       for (x, y, w, h) in faces:
            center = [x+w/2,y+h/2,1,1] #center=[axis_x,axis_y,hit,counter]
            index = find_nearest_center(center)
            # mark the number of face 
            # cv2.putText(gray,text=str(index),org=(x,y),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255,255,255))
            if index != -1:
                centers[index][2] += 1 #hit
                #update centers[index]
                centers[index][0] = (centers[index][0]+center[0])/2.0
                centers[index][1] = (centers[index][1]+center[1])/2.0
                if index==0: #confine to the first face
                    if centers[index][2] >= 30:
                        cv2.rectangle(gray, (x, y), (x+w, y+h), 0, 2)
                        global_face_roi = gray[y:y+h,x:x+w]
                        #cv2.imshow("face",face_roi)
                        #detection session
                        eyes = eyeCascade.detectMultiScale(
                            global_face_roi,
                            scaleFactor=1.1,
                            minNeighbors=3,
                            minSize=(1,1),
                            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                        )
                        h_ratio = float(h)/640
                        w_ratio = float(w)/480
                        #enlarge the region of face to size of (640,480)
                        local_face_roi = cv2.resize(global_face_roi,(640,480),interpolation=cv2.INTER_AREA)
                        mouths = mouthCascade.detectMultiScale(
                            local_face_roi,
                            scaleFactor=1.5,
                            minNeighbors=1,
                            minSize=(1,1),
                            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                        )
                                
                        for tc, content in enumerate(zip(mouths,eyes)):
                            mx,my,mw,mh = content[0]
                            ex,ey,ew,eh = content[1]
                            if my+mh/2 > 320 and mw<200 and mh <200: #filter for mouth
                                if ex+eh/2 <320 and ew<200 and eh<200:#filter for eye
                                   #crop the mouth in the local image and resize the cropped mouth to fixed size
                                   crop_mouth = local_face_roi[my:my+mh,mx:mx+mw]
                                   crop_mouth = cv2.resize(crop_mouth,(20,20))
                                   #crop the eye in the global image and resize the cropped eye to fixed size
                                   crop_eye = global_face_roi[ey:ey+eh,ex:ex+ew]
                                   crop_eye = cv2.resize(crop_eye,(20,10))
                                   #imwrite crop mouth
                                   cv2.imwrite("G:\\vs2010\\59NetWork\\identity_state_project\\data\\mouth_data\\"+str(counter)+".bmp",crop_mouth)
                                   cv2.imwrite("G:\\vs2010\\59NetWork\\identity_state_project\\data\\eye_data\\"+str(counter)+".bmp",crop_eye)
                                   counter += 1
                   
                   #if mean_value[index][2] > 2:
                   #    cv2.putText(gray,"speaking!",(x+w,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,55)

            else: #if index!=-1
                centers.append(center)
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    else: #if cap.read()
        break

cap.release()
#out.release()
cv2.destroyAllWindows()