# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 18:32:21 2021

@author: luc40
"""
#import function_head as H
import cv2
import numpy as np
from cv2 import fitEllipse
# =============================================================================
# from skimage.feature import canny
# from skimage.transform import hough_ellipse
# from scipy.signal import find_peaks, peak_widths
# =============================================================================
from matplotlib import pyplot as plt

def get_eclipse_param_pupil(params):  
    params.filterByColor = False
    params.minThreshold = 65
    params.maxThreshold = 93
    params.blobColor = 225
    params.minArea = 150
    params.maxArea = 6000
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.minCircularity =.95
    params.maxCircularity = 1
    return params

def get_eclipse_param_iris(params):  
    params.filterByColor = False
    params.minThreshold = 65
    params.maxThreshold = 93
    params.blobColor = 225
    params.minArea = 1000
    params.maxArea = 6000
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.minCircularity =.9
    params.maxCircularity = 1
    return params

def getCircles(image):
	i = 150; j = 80
	while j >30:
		circles = cv2.HoughCircles(
             image,cv2.HOUGH_GRADIENT,
             dp=2,
             minDist=300,
             param1=i,
             param2=j ,minRadius=35,maxRadius=50)
		if circles is not None:
			return circles
		i -= 1; j -= 1
	return ([])

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    import math

    brightness = 0
    contrast = +100 # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def createEyeMask(eyeLandmarks, im):
    leftEyePoints = eyeLandmarks
    eyeMask = np.zeros_like(im)
    cv2.fillConvexPoly(eyeMask, np.int32(leftEyePoints), (255, 255, 255))
    eyeMask = np.uint8(eyeMask)
    return eyeMask

def capture_eye_pupil(frame,eyes):
    OD_p = []; OS_p = []; thr_eyes = []
       
    #frame = cv2.inpaint(frame,frame_blr,3,cv2.INPAINT_TELEA)
    for (ex,ey,ew,eh) in eyes:  
        roi_color2 = frame[ey:ey+eh, ex:ex+ew]   
        gray = cv2.cvtColor(roi_color2, cv2.COLOR_BGR2GRAY)        
        gray = modify_contrast_and_brightness2(gray)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)        
# =============================================================================
#         cv2.destroyAllWindows()
# =============================================================================
        GET_CIRCLE = False;
# =============================================================================
#         a = np.where(gray<=minVal)
#         kernal = (int(np.percentile(a[0],50)),int(np.percentile(a[1],50)))
# =============================================================================
        kernal =(minLoc[1],minLoc[0])     
        pre_thr_xL = gray[kernal]
        pre_thr_xR = gray[kernal]
        pre_thr_yD = gray[kernal]
        pre_thr_yU = gray[kernal]
        i = 0; dxL = 0; dxR = 0; dyD = 0; dyU = 0
        STOP_L = False; STOP_R = False; STOP_D = False; STOP_U = False;
        while i <= 75:
            x_2L = kernal[0]+dxL
            x_2R = kernal[0]-dxR
            y_2D = kernal[1]+dyD
            y_2U = kernal[1]-dyU
            try:
                thr_xL = gray[x_2L,kernal[1]]
                thr_xR = gray[x_2R,kernal[1]]
                thr_yD = gray[kernal[0],y_2D]
                thr_yU = gray[kernal[0],y_2U]
                if ((thr_xL-pre_thr_xL>15 or thr_xL>90) and 
                    (thr_xR-pre_thr_xR>15 or thr_xR>90) and
                    (thr_yD-pre_thr_yD>15 or thr_yD>90) and
                    (thr_yU-pre_thr_yU>15 or thr_yU>90)): 
                    thr = np.mean([pre_thr_xL,pre_thr_xR,pre_thr_yD,pre_thr_yU])
                    break
                
                if not (thr_xL-pre_thr_xL>15 or thr_xL>90) and not STOP_L: dxL = i+2
                else: STOP_L = True
                
                if not (thr_xR-pre_thr_xR>15 or thr_xR>90) and not STOP_R: dxR = i+2
                else: STOP_R = True
                
                if not (thr_yD-pre_thr_yD>15 or thr_yD>90) and not STOP_D: dyD = i+2
                else: STOP_D = True
                
                if not (thr_yU-pre_thr_yU>15 or thr_yU>90) and not STOP_U: dyU = i+2
                else: STOP_U = True
                
                if thr_xL > pre_thr_xL and thr_xL < 90: pre_thr_xL = thr_xL
                if thr_xR > pre_thr_xR and thr_xR < 90: pre_thr_xR = thr_xR
                if thr_yD > pre_thr_yD and thr_yD < 90: pre_thr_yD = thr_yD
                if thr_yU > pre_thr_yU and thr_yU < 90: pre_thr_yU = thr_yU                
                thr = np.mean([pre_thr_xL,pre_thr_xR,pre_thr_yD,pre_thr_yU])
            except:
                thr = np.mean([pre_thr_xL,pre_thr_xR,pre_thr_yD,pre_thr_yU])
                break
            i+=2
            
            
            
        thr = int(thr)
        if thr>=180: thr = 90
        while not GET_CIRCLE and thr<180:            
            _,roi_gray1 = cv2.threshold(gray,thr,255,0)
# =============================================================================
#             cv2.imshow('gray',gray) 
#             cv2.imshow('roi_gray1',roi_gray1) 
#             cv2.waitKey(1) 
# =============================================================================
            #print(thr)
            params_p = cv2.SimpleBlobDetector_Params()
            params_p = get_eclipse_param_pupil(params_p)
            det = cv2.SimpleBlobDetector_create(params_p)
            circles = det.detect(roi_gray1)            
            OD_pre_size = 0; OS_pre_size = 0; OD_ds_pre = 1000; OS_ds_pre = 1000;
            if len(circles)>0:  
                GET_CIRCLE = True
                for kp in circles:
                    distance = np.linalg.norm(np.array(kernal)-np.array([kp.pt[0],kp.pt[1]]))
                    #print('distance: '+str(distance))
                    if ew>=274:
                        if ex == eyes[0][0] and int(kp.size/2)>OD_pre_size:                        
                            OD_pre_size = int(kp.size/2)
                            OD_p = [int(kp.pt[0]+eyes[0][0]), int(kp.pt[1]+eyes[0][1]), int(kp.size/2)]
                        elif ex == eyes[1][0] and int(kp.size/2)>OS_pre_size:
                            OS_pre_size = int(kp.size/2)
                            OS_p = [int(kp.pt[0]+eyes[1][0]), int(kp.pt[1]+eyes[1][1]), int(kp.size/2)]
                    else:
                        if ex == eyes[0][0] and distance < OD_ds_pre:   
                            OD_ds_pre = distance
# =============================================================================
#                             if distance>35:
#                                 OD_p = [int(kernal[0]+eyes[0][0]), int(kernal[1]+eyes[0][1]), int(kp.size/2)]
#                             else:
# =============================================================================
                            OD_p = [int(kp.pt[0]+eyes[0][0]), int(kp.pt[1]+eyes[0][1]), int(kp.size/2)]
                        elif ex == eyes[1][0] and distance < OS_ds_pre:
                            OS_ds_pre = distance
# =============================================================================
#                             if distance>35:
#                                 OS_p = [int(kernal[0]+eyes[1][0]), int(kernal[1]+eyes[1][1]), int(kp.size/2)]                
#                             else:
# =============================================================================
                            OS_p = [int(kp.pt[0]+eyes[1][0]), int(kp.pt[1]+eyes[1][1]), int(kp.size/2)]                
            else:
                thr += 2
                if ex == eyes[0][0]:
                    OD_p=[np.nan,np.nan,np.nan]
                elif ex == eyes[1][0]:
                    OS_p=[np.nan,np.nan,np.nan]
        thr_eyes.append(thr)
    return np.array(OD_p),np.array(OS_p),np.array(thr_eyes)

def capture_eye_iris(frame,eyes):
    OD_p = []; OS_p = []; thr_eyes = []
       
    #frame = cv2.inpaint(frame,frame_blr,3,cv2.INPAINT_TELEA)
    for (ex,ey,ew,eh) in eyes:  
        roi_color2 = frame[ey:ey+eh, ex:ex+ew]   
        gray = cv2.cvtColor(roi_color2, cv2.COLOR_BGR2GRAY)        
        gray = modify_contrast_and_brightness2(gray)
        gray = cv2.medianBlur(gray,15)         
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)        
# =============================================================================
#         cv2.destroyAllWindows()
# =============================================================================
        GET_CIRCLE = False; 
        pre_thr = 125; i = 0;
        while i <= 55:
            try:
                thr_x = gray[np.array(minLoc)[1]+20+i,np.array(minLoc)[0]]
                thr_y = gray[np.array(minLoc)[1],np.array(minLoc)[0]+20+i]
                if thr_x-pre_thr>60 and thr_y-pre_thr>60: thr = pre_thr+10; break
                elif thr_x < thr_y: pre_thr = thr_x
                elif thr_y < thr_x: pre_thr = thr_y
                thr = pre_thr+10;
            except:
                thr = pre_thr+10; break
            i+=2

        while not GET_CIRCLE and thr<180:            
            _,roi_gray1 = cv2.threshold(gray,thr,thr,cv2.THRESH_BINARY_INV)
            cv2.imshow('gray',gray) 
            cv2.imshow('roi_gray1',roi_gray1) 
            cv2.waitKey(1) 
            #print(thr)
            params_p = cv2.SimpleBlobDetector_Params()
            params_p = get_eclipse_param_iris(params_p)
            det = cv2.SimpleBlobDetector_create(params_p)
            circles = det.detect(roi_gray1)            
            OD_pre_size = 0; OS_pre_size = 0; OD_ds_pre = 1000; OS_ds_pre = 1000;
            if len(circles)>0:  
                GET_CIRCLE = True
                for kp in circles:
                    distance = np.linalg.norm(np.array(minLoc)-np.array([kp.pt[1],kp.pt[0]]))
                    #print('distance: '+str(distance))
                    if ew>=274:
                        if ex == eyes[0][0] and int(kp.size/2)>OD_pre_size:                        
                            OD_pre_size = int(kp.size/2)
                            OD_i = [int(kp.pt[0]+eyes[0][0]), int(kp.pt[1]+eyes[0][1]), int(kp.size/2)]
                        elif ex == eyes[1][0] and int(kp.size/2)>OS_pre_size:
                            OS_pre_size = int(kp.size/2)
                            OS_i = [int(kp.pt[0]+eyes[1][0]), int(kp.pt[1]+eyes[1][1]), int(kp.size/2)]
                    else:
                        if ex == eyes[0][0] and OD_ds_pre>distance:   
                            #print('OD: '+str(distance))
                            OD_ds_pre = distance
                            OD_i = [int(kp.pt[0]+eyes[0][0]), int(kp.pt[1]+eyes[0][1]), int(kp.size/2)]
                        elif ex == eyes[1][0] and OS_ds_pre>distance:
                            #print('OS: '+str(distance))
                            OS_ds_pre = distance
                            OS_i = [int(kp.pt[0]+eyes[1][0]), int(kp.pt[1]+eyes[1][1]), int(kp.size/2)]                
            else:
                thr += 5
                if ex == eyes[0][0]:
                    OD_i=[np.nan,np.nan,np.nan]
                elif ex == eyes[1][0]:
                    OS_i=[np.nan,np.nan,np.nan]
        thr_eyes.append(thr)
    return np.array(OD_i),np.array(OS_i),np.array(thr_eyes)

def get_eye_position(cap,eyes):
    frame_count = 0; OD = []; OS = []
    
    #cap = GetVideo(Gaze9_Task.csv_path)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame_count < 60 and ret:
            frame_count+=1                        
            OD_p,OS_p,thr = capture_eye_pupil(frame,eyes)
            OD.append(OD_p)
            OS.append(OS_p)            
        else:
            break   
        
    OD = np.array(OD); OS = np.array(OS)
    if len(np.argwhere(np.isnan(OD)))<len(OD)*3:
        OD_eye = np.nanpercentile(OD,50,axis = 0)
    else:
        OD_eye = [eyes[0][0], eyes[0][1], 0]
    if len(np.argwhere(np.isnan(OS)))<len(OS)*3:
        OS_eye = np.nanpercentile(OS,50,axis = 0)
    else:
        OS_eye = [eyes[1][0], eyes[1][1], 0]
    eyes = [[int(OD_eye[0]-100),int(OD_eye[1]-75),200,150],
            [int(OS_eye[0]-100),int(OS_eye[1]-75),200,150]]      
    return np.abs(eyes), OD_eye, OS_eye

#cap = GetVideo(ACT_Task.csv_path)
