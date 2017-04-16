########################################################
################# Using Graph Cut ######################
########################################################
import numpy as np
import cv2
from skimage import segmentation,color
from skimage.future import graph
from matplotlib import pyplot as plt
import scipy.io as sio

videoPath = "G:\\vs2010\\59NetWork\\video\\IMG_2269.MOV"
cap = cv2.VideoCapture(videoPath)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# get frame rate
if int(major_ver) < 3:
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print "Frames per second using video.get(cv2.CAP_PROP_FPS): {0}".format(fps)

# set the duration(/s) of taking average
duration = 5
frames_for_every_seconds = 3
every_seconds_counter = 0
# work out frames needed to be average
fps_counter = 0
loop_counter_down = duration*fps

total_stack_frames = duration * frames_for_every_seconds
stack_frame_counter = 0
stack_frames_variation = np.zeros((total_stack_frames-2,))

width = cap.get(3)
height = cap.get(4)
width /= 4
height /= 4
width = int(width)
height = int(height)
frame_stack = np.zeros((int(height),int(width),int(total_stack_frames+1)),dtype=float)

#for super pixels division
compactness_value = 10
n_segments_value = 150

index = 0 #for save avg images


standard_label = np.zeros((int(height),int(width)))
first_time = True

def label_alignment(area1,area2):
    #align the lable of area2 to area1
    #area1 and area2 is numpy array
    max_1 = np.amax(area1)
    max_2 = np.amax(area2)
    min_overlapped = 90
    for i in xrange(max_2):
        correspond_index = i
        max_overlapped = 0
        index_of_i = np.ravel_multi_index(np.where(area2==i),area1.shape)
        for j in xrange(max_1):
           index_of_j = np.ravel_multi_index(np.where(area1==j),area2.shape)
           overlapped = np.intersect1d(index_of_i,index_of_j,True).shape
           if overlapped > min_overlapped:
                if overlapped > max_overlapped:
                    max_overlapped = overlapped
                    correspond_index = j
        area2[area2==i] = correspond_index

while(cap.isOpened() and loop_counter_down>=0):
    ret,frame = cap.read()
    loop_counter_down -= 1
    if ret == True:
        fps_counter += 1
        fps_counter = fps_counter%int(fps)
        #read every second
        if  fps_counter == 1:  
            for every_seconds_counter in xrange(frames_for_every_seconds):
                frame = cv2.resize(frame,(int(width),int(height)),interpolation = cv2.INTER_CUBIC)
                #segment it into super pixels
                labels1 = segmentation.slic(frame,compactness = compactness_value, n_segments = n_segments_value)
                out1 = color.label2rgb(labels1,frame,kind = 'avg') #out1 is the result of super pixel
                #construct region adjacency graph
                g = graph.rag_mean_color(frame, labels1, mode='similarity')
                labels2 = graph.cut_normalized(labels1,g)
                labels2 = np.asarray(labels2)
                #perform label alignment
                if not first_time:
                    label_alignment(standard_label,labels2)

                first_time = False
                standard_label = labels2
                #out2 = color.label2rgb(labels2,frame, kind = 'avg') #out2 is the classfied result of graph cut
                sio.savemat("G:\\vs2010\\59NetWork\\video\\test_labels"+str(stack_frame_counter)+".mat",mdict={'labels':labels2})
                frame_stack[:,:,stack_frame_counter] = labels2
                
                stack_frame_counter =(stack_frame_counter)%total_stack_frames
                if stack_frame_counter==total_stack_frames-1:  #averaging
                    avg_image = np.mean(frame_stack,axis = 2)
                    cv2.imwrite("G:\\vs2010\\59NetWork\\video\\avg_experiment_"+str(index)+".png",avg_image)
                    for k in xrange(1,total_stack_frames-1):
                        stack_frames_variation[k-1] = np.sum((frame_stack[:,:,k-1] - frame_stack[:,:,k]) + (frame_stack[:,:,k+1] - frame_stack[:,:,k]))
                        stack_frames_variation[k-1] /= int(width)*int(height)
                        frame_stack[:,:,k] -= avg_image
                    index += 1
                    np.save("G:\\vs2010\\59NetWork\\video\\frame_stack_variation.npy",stack_frames_variation)
                    np.save("G:\\vs2010\\59NetWork\\video\\frame_stack.npy",frame_stack)
                stack_frame_counter += 1

                #continue to read frame
                ret,frame = cap.read()
                loop_counter_down -= 1
                fps_counter += 1