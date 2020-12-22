#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return [] 
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print help_message

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")
    
    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')
    
    first = True
    firstDetected = True
    width = 0
    hight = 0 
    w = 0
    h = 0
    
    # Create some random colors
    col = np.random.randint(0,255,(100,3))
   
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        facesDetected = detect(gray, cascade)
        vis = img.copy()
        
        #TO DO: a few faces        
        rects = np.ndarray(shape=(2,4),dtype = np.int32)

        if facesDetected != [ ]:
            for face in facesDetected:
                [x1,y1,x2,y2] = face 
                print(first)
                if first:
                    width = x2 - x1
                    hight = y2 - y1 
                    w = np.int32(((0.5*width)/2))
                    h = np.int32(((0.1*hight)/2))
                    width = width -2*w
                    hight = hight -2*h
                    S1 = np.int32( hight*0.45) 
                    S2 = np.int32(hight*0.2)
                    first = False
                    firstDetected = True
                rects[0][0] = x1 + w
                rects[0][1] = y1 + h + S1  
                rects[0][2] = x1 + w + width
                rects[0][3] = y1 + h + hight

                rects[1][0] = x1 + w
                rects[1][1] = y1 + h 
                rects[1][2] = x1 + w + width
                rects[1][3] = y1 + h + S2
            draw_rects(vis, rects, (0, 255, 0))
        else:
            print("Any face isn't detected")
		
        #Lucas Kanade tracer
        #error when program lost the face
        '''[x11, y11, x12, y12 ] = rects[0]
        [x21, y21, x22, y22 ] = rects[1]
        vis_roi1 = vis[y11:y12, x11:x12]
        vis_roi2 = vis[y21:y22, x21:x22]
        if firstDetected:
            old_gray1 = cv2.cvtColor(vis_roi1, cv2.COLOR_BGR2GRAY)
            p1_old = (cv2.goodFeaturesToTrack(old_gray1, mask = None, **feature_params))
            old_gray2 = cv2.cvtColor(vis_roi2, cv2.COLOR_BGR2GRAY)
            p2_old = (cv2.goodFeaturesToTrack(old_gray2, mask = None, **feature_params))
            firstDetected = False
            mask = np.zeros_like(vis_roi1)
        else:
            frame_gray1 = cv2.cvtColor(vis_roi1, cv2.COLOR_BGR2GRAY)
            frame_gray2 = cv2.cvtColor(vis_roi2, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray1, frame_gray1, p1_old, None, **lk_params) 
            p2, st, err = cv2.calcOpticalFlowPyrLK(old_gray2, frame_gray2, p2_old, None, **lk_params)
            
            #good_new1 = p1[st==1]
            #good_old1 = p1_old[st==1]   
            print(p1)
            #good_new2 = p2[st==1]
            #good_old2 = p2[st==1]

            for i,new in enumerate(p1):
                a,b = new.ravel()
                #mask = cv2.line(mask, (a,b), col[i].tolist(), 2)
                vis_roi1 = cv2.circle(vis_roi1,(a,b),3,col[i].tolist(),-1)
            img = cv2.add(vis_roi1,mask)
            if img!= None:
                cv2.imshow('frame',img)'''
            
            # Now update the previous frame and previous points
            #old_gray1 = frame_gray1.copy()
            #p1 = p1.reshape(-1,1,2)
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break


    cv2.destroyAllWindows()
