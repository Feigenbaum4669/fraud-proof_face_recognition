#!/usr/bin/env python

import numpy as np
import cv2
import cv2 as cv
from video import create_capture
from common import clock, draw_str
import time
import os

#todo: general parameters
#todo: error:lost point
#todo: parameter optimialization
#todo: #faces >1
#todo: visualization

# params for ShiTomasi corner detection
# todo docker
feature_params = dict( maxCorners = 500,
    qualityLevel = 0.001,
    minDistance = 10.0,
    blockSize = 3,
    useHarrisDetector = 0,
    k = 0.4 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (25,25),
    maxLevel = 5,
    flags = 0,
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.0001))

# parameters for face detector
fd_params = dict( scaleFactor = 1.1,
    minNeighbors = 4,
    minSize = (30,30),
    flags = cv.CASCADE_SCALE_IMAGE )

def detect(img, cascade_fn= "data/haarcascades/haarcascade_frontalface_alt.xml"):
    
    cascade = cv2.CascadeClassifier(cascade_fn)
    faces = cascade.detectMultiScale(img, **fd_params)
    if len(faces) == 0:
        return [] 
    faces[:,2:] += faces[:,:2]
    
    #TO DO: a few faces        
    rects = np.ndarray(shape=(2,4),dtype = np.int32)
    
    for face in faces:
        [x1,y1,x2,y2] = face 
        #if first:
        #detected = True
        width = x2 - x1
        height = y2 - y1 
        w = np.int32(((0.5*width)/2))
        h = np.int32(((0.1*height)/2))
        width = width -2*w
        height = height -2*h
        S1 = np.int32( height*0.45) 
        S2 = np.int32(height*0.2)
        #first = False
        #firstDetected = True
        
        rects[0][0] = x1 + w
        rects[0][1] = y1 + h + S1  
        rects[0][2] = x1 + w + width
        rects[0][3] = y1 + h + height
        rects[1][0] = x1 + w
        rects[1][1] = y1 + h 
        rects[1][2] = x1 + w + width
        rects[1][3] = y1 + h + S2

    return rects

#Lucas Kanade tracer
def good_points_to_track(rects,gray):
    
    #mayby can be usefull sth more general
    
    [x11, y11, x12, y12 ] = rects[0]
    [x21, y21, x22, y22 ] = rects[1]    
    Rmask = np.zeros(gray.shape,np.uint8)
    Rmask[y11:y12,x11:x12] = gray[y11:y12,x11:x12]
    p1 = (cv2.goodFeaturesToTrack(Rmask, mask = None, **feature_params))
    Rmask = np.zeros(gray.shape,np.uint8)
    Rmask[y21:y22,x21:x22] = gray[y21:y22,x21:x22]
    p2 = (cv2.goodFeaturesToTrack(Rmask, mask = None, **feature_params))
    p = [p1, p2]
    
    return p

def tracking(video_src=0):

    cam = create_capture(video_src)
    detected = False
    firstDetected = False
    fs = 60 # Hz
    T = 1.0 / fs #ms
    for t in range(fs*5): # bylo fs = 25
        time.sleep(T)
        cam.set(cv2.CAP_PROP_POS_MSEC,t*T)
        try:
            ret, img = cam.read()
        except:
            print("error")
            break
        cv2.imwrite('img{}.jpg'.format(t),img)
	
	PointSeries = []
	
    for tt in range(t):
        try:
            img = cv2.imread('img{}.jpg'.format(tt))
        except:
            break
        img = cv2.resize(img, (640, 310))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detect(gray)
        vis = img.copy()
        
        if rects != [ ]:
            if not detected:
                firstDetected = True
        else:
            pass

        if firstDetected:
            old_gray = gray.copy()
            p_old = good_points_to_track(rects,gray)
            p = [0,0]
            firstDetected = False
            detected = True
        elif detected:
            # calculate optical flow
            p[0], st0, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p_old[0], None, **lk_params) 
            p[1], st1, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p_old[1], None, **lk_params)
            print(len(p[0])+len(p[1]))
            try:
                good_new1 = p[0][st0==1]
                good_old1 = p_old[0][st0==1]   
                good_new2 = p[1][st1==1]
                good_old2 = p_old[1][st1==1]
            except:
                detected = False
                firstDetected = False
            
            #Now update the previous frame and previous points
            old_gray = gray.copy()
            p_old[0] = good_new1.reshape(-1,1,2)
            p_old[1] = good_new2.reshape(-1,1,2)
            
            points = []
            for px1 in good_new1:
                 (a,b) = px1.ravel()
                 points.append((a,b))

            for px2 in good_new2:
                 (a,b) = px2.ravel()
                 points.append((a,b))
            
            #print(points)
            print("p")
            PointSeries.append([tt*T, points])
        
        os.remove('img{}.jpg'.format(tt))
	
    return(PointSeries)

 	
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys
    video_src = sys.argv[1:]
    
    try: video_src = video_src[0]
    except: video_src = 0

    cam = create_capture(video_src)
    print(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    detected = False
    firstDetected = False
    
    #random color to visualization "good points"
    color = np.random.randint(0,255,(500,3))
    counter = 0
    while True:

        try:
            ret, img = cam.read()
        except:
            break
        
        img = cv2.resize(img, (1280, 620)) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        t = clock()
        rects = detect(gray)
        vis = img.copy()

        if rects != [ ]:
            if not detected:
                firstDetected = True
            draw_rects(vis, rects, (0, 255, 0))
        else:
            print("Any face isn't detected")

        if firstDetected:
            old_gray = gray.copy()
            p_old = good_points_to_track(rects,gray)
            p = [0,0]
            firstDetected = False
            detected = True
        elif detected:
            # calculate optical flow
            p[0], st0, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p_old[0], None, **lk_params) 
            p[1], st1, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p_old[1], None, **lk_params)
            print(len(p[0])+len(p[1]))
            try:
                good_new1 = p[0][st0==1]
                good_old1 = p_old[0][st0==1]   
                good_new2 = p[1][st1==1]
                good_old2 = p_old[1][st1==1]
            except:
                detected = False
                firstDetected = False
            
            for i,(new,old) in enumerate(zip(good_new1,good_old1)):
                a,b = new.ravel()
                cv2.circle(vis,(a,b),5,color[i].tolist(),-1)
            for i,(new,old) in enumerate(zip(good_new2,good_old2)):
                a,b = new.ravel()
                cv2.circle(vis,(a,b),5,color[i].tolist(),-1)
            #Now update the previous frame and previous points"""
            old_gray = gray.copy()
            p_old[0] = good_new1.reshape(-1,1,2)
            p_old[1] = good_new2.reshape(-1,1,2)
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        
        cv2.imshow('frame',vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break


    cv2.destroyAllWindows()
	
