import cv2
import numpy as np
import os
from Trackers import *

        
def getTracker(cap,objs,tracker_type):
    
    if tracker_type == 1: #KLT
        KLTTracking(cap,objs,play_realtime=True,save_to_file=False)
        
        print("Tracking done")
        
    if tracker_type == 2: #MST
        MSTracking(cap,objs,play_realtime=True,save_to_file=False)
        print("Tracking done")
        
    if tracker_type == 3:
        tracker = cv2.TrackerBoosting_create()
        print("Tracker created")
        Tracker(cap,tracker,objs)
        print("Tracking done")
        
    if tracker_type == 4:
        tracker = cv2.TrackerMIL_create()
        Tracker(cap,tracker,objs)
        print("Tracking done")
        
    if tracker_type == 5:
        tracker = cv2.TrackerKCF_create()
        Tracker(cap,tracker,objs)
        print("Tracking done")
        
    if tracker_type == 6:
        tracker = cv2.TrackerTLD_create()
        Tracker(cap,tracker,objs)
        print("Tracking done")

    if tracker_type == 7:
        tracker = cv2.TrackerMedianFlow_create()
        Tracker(cap,tracker,objs)
        print("Tracking done")
        
    if tracker_type == 8:
        tracker = cv2.TrackerGOTURN_create()
        Tracker(cap,tracker,objs)
        print("Tracking done")
        
    if tracker_type == 9:
        tracker = cv2.TrackerMOSSE_create()
        Tracker(cap,tracker,objs)
        print("Tracking done")
        
    if tracker_type == 10:
        tracker = cv2.TrackerCSRT_create()
        Tracker(cap,tracker,objs)
        print("Tracking done")
        
if __name__ == "__main__":
    
    path = '/home/gokul/Documents/Studies/TrackaROI/Video.mp4'
    cap = cv2.VideoCapture(path)
    ret,frame = cap.read()
    if not ret:
        raise ("No video read")
        cap.release()
    
    n_object = int(input("Number of objects to track:"))
    objs = np.empty((n_object,4), dtype=float)

    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frame)
        cv2.destroyWindow("Select Object %d"%(i))
        print(objs.shape)
        objs[i,:] = (xmin, ymin, boxw, boxh)
        objs[i,:] = objs[i,:].astype(float)
#         objs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]])
    cap.release()
    del cap
    
    tracker = {1: 'KLT',2:'MeanShift',3: 'BOOSTING',4:'MIL',5: 'KCF',6:'TLD',7:'MEDIANFLOW',8:'GOTURN',9:'MOSSE',10:'CSRT'}
    print(tracker)
    tracker_type = int(input("Choose the tracker:"))
    frames = []    
    
    video = cv2.VideoCapture(path)
    while True:
        ret, f = video.read()
        if ret:
            frames.append(np.array(f))
        else:
            print("Nothing read")
            break
    print(len(frames),frames[20].shape,objs.shape)
    
    getTracker(frames,objs,tracker_type)
    
    