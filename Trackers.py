import numpy as np
import cv2
import sys
import os
from helper_functions import *

def MSTracking(frames, objs, play_realtime=True, save_to_file=False):
    # initialise:
    n_frame = len(frames)
    n_obj = len(objs)    
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)    
    
    if save_to_file:
        out = cv2.VideoWriter('MSToutput.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(frames[0].shape[1],frames[0].shape[0]))
    
    roi_hist = []
    if n_obj>1:
        for i in range(n_obj):
            (c,r,w,h) = objs[i].astype(int)
            # set up the ROI for tracking
            roi = frames[0][r:r+h, c:c+w]
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist.append(cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]))
            cv2.normalize(roi_hist[i],roi_hist[i],0,255,cv2.NORM_MINMAX)
    else:
        (c,r,w,h) = objs[0].astype(int)
        # set up the ROI for tracking
        roi = frames[0][r:r+h, c:c+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist[i],roi_hist[i],0,255,cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    for i in range(n_frame):        
        hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)        
        frames_draw[i] = frames[i].copy()
        if n_obj>1:
            for j in range(n_obj):
                (c,r,w,h) = objs[j].astype(int)
    #             print(c,r,w,h)
                dst = cv2.calcBackProject([hsv],[0],roi_hist[j],[0,180],1)
                ret, track_window = cv2.meanShift(dst, (c,r,w,h), term_crit)
                # Draw it on image
                x,y,w,h = track_window
                frames_draw[i] = cv2.rectangle(frames_draw[i], (x,y), (x+w,y+h), 255,2)
        else:
            (c,r,w,h) = objs[0].astype(int)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret, track_window = cv2.meanShift(dst, (c,r,w,h), term_crit)
            # Draw it on image
            x,y,w,h = track_window
            frames_draw[i] = cv2.rectangle(frames_draw[i], (x,y), (x+w,y+h), 255,2)    
    print("Done Processing")
 
        if play_realtime:
            for i in range(n_frame):     
                cv2.imshow("Visualise",frames_draw[i])
                cv2.waitKey(20)
        if save_to_file:
                out.write(frames_draw[i])

        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    if save_to_file:
        out.release()


def KLTTracking(frames, objs, play_realtime=True, save_to_file=False):
    # initialization
    n_frame = len(frames)
    n_object = len(objs)
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    bboxs[0] = np.empty((n_object,4,2), dtype=float)
    
    for i in range(n_object):        
        (xmin, ymin, boxw, boxh) = objs[i]
        bboxs[0][i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
    print(bboxs[0].shape,len(bboxs),n_object)
    
    
    if save_to_file:
        out = cv2.VideoWriter('output.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(frames[0].shape[1],frames[0].shape[0]))
    
    # Start from the first frame, do optical flow for every two consecutive frames.
    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_RGB2GRAY),bboxs[0],use_shi=False) # obtains harris corner interest points as descriptors.
    
    for i in range(1,n_frame):
        print('Processing Frame in pairs',i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i]) # compute the transformation between the two images
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1]) # estimate the new X,Y and bounding box positions by previous and new features.
        
        # update coordinates
        startXs = Xs
        startYs = Ys
        # update feature points as required
        n_features_left = np.sum(Xs!=-1)
        print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_RGB2GRAY),bboxs[i])

        # draw bounding box and visualize feature point for each object
        frames_draw[i] = frames[i].copy()
        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][j,:,:].astype(int))
            frames_draw[i] = cv2.rectangle(frames_draw[i], (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
            for k in range(startXs.shape[0]):
                frames_draw[i] = cv2.circle(frames_draw[i], (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)
        
        # imshow if to play the result in real time
        if play_realtime:
            cv2.imshow("Visualise",frames_draw[i])
            cv2.waitKey(10)
        if save_to_file:
            out.write(frames_draw[i])
        
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
    
    if save_to_file:
        out.release()
        
def Tracker(frames,tracker,objs,play_realtime=True,save_to_file=False):
    n_obj = len(objs)
    n_frame = len(frames)
#     bboxs = np.empty((n_frame,),dtype=np.ndarray)
    
    if save_to_file:
        out = cv2.VideoWriter('output.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(frames[0].shape[1],frames[0].shape[0]))
    
    multiTracker = cv2.MultiTracker_create() 
    for bbox in objs:
        (c,r,w,h) = bbox.astype(int)
        multiTracker.add(tracker, frames[0], (c,r,w,h))
        
    for i in range(n_frame):
        success,boxes = multiTracker.update(frames[i])
         # draw tracked objects     
        for _,newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frames[i], p1, p2, (255,0,0), 2, 1)
        
        # imshow if to play the result in real time
        if play_realtime:
            cv2.imshow("Visualise",frames[i])
            cv2.waitKey(10)
        
        if save_to_file:
            out.write(frames[i])
        
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
            
    if save_to_file:
        out.release()
            
        
 
