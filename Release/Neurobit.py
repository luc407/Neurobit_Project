# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:35:56 2022

@author: luc40
"""
import glob
import os
import math
import cv2
import numpy as np
import pandas as pd 
import qrcode
import sqlite3
import time
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip, concatenate_videoclips
from datetime import datetime
from function_eye_capture import capture_eye_pupil

EYE         = ['OD','OS']
ACT         = 'ACT'
CUT         = 'CUT'
GAZE_9      = '9_Gaze'

DX          = np.array(["control","horizontal strabismus","vertical strabismus",
               "nystagmus","orbital frx","nerve palsy","TAO"])

ACT_TIME    = ["O_t","CL_t", "CR_t",  "UCR_t"]
ACT_COLOR   = ['aqua','plum','gold','lime']
ACT_STR     = ["Open", "Cover Left", "Cover Right", "Uncover"]
ACT_LABEL   = ['O','CL','CR','UCR']

CUT_TIME    = ["O_t", "CL_t", "UCL_t", "CR_t",  "UCR_t"]
CUT_STR     = ["Open", "Cover Left", "Uncover Left", "Cover Right", "Uncover Right"]
CUT_LABEL   = ['O','CL','UCL','CR','UCR']

GAZE_9_TIME     = ['F','U','D','R','L','RU','LU','RD','LD']
GAZE_9_COLOR    = ['deepskyblue',  'wheat',    'mediumpurple', 
                   'slateblue',    'plum',     'lime',         
                   'aqua',         'gold',     'orange']
GAZE_9_STR      = ["Front",       "Up",    "Down",     
                   "Right",  "Left",  "Right Up",  
                   "Left Up", "Right Down", "Left Down"]

GAZE_9_EYEFIG = np.array([])
GAZE_9_BOARDER = np.ones(8).astype(int)
for T in range(0,len(GAZE_9_TIME)):
    if GAZE_9_TIME[T] == 'F':
        GAZE_9_EYEFIG = np.array([5])
    elif GAZE_9_TIME[T] == 'U':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[2]))
        GAZE_9_BOARDER[2] = T
    elif GAZE_9_TIME[T] == 'D':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[8]))
        GAZE_9_BOARDER[6] = T
    elif GAZE_9_TIME[T] == 'R':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[4]))
        GAZE_9_BOARDER[0] = T
    elif GAZE_9_TIME[T] == 'L':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[6]))
        GAZE_9_BOARDER[4] = T
    elif GAZE_9_TIME[T] == 'RU':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[1]))
        GAZE_9_BOARDER[1] = T
    elif GAZE_9_TIME[T] == 'LU':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[3]))
        GAZE_9_BOARDER[3] = T
    elif GAZE_9_TIME[T] == 'RD':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[7]))
        GAZE_9_BOARDER[7] = T
    elif GAZE_9_TIME[T] == 'LD':
        GAZE_9_EYEFIG = np.concatenate((GAZE_9_EYEFIG,[9]))
        GAZE_9_BOARDER[5] = T

# color map
line_color_palatte = {'greens':["#A5F5B3", "#51F46D",   "#00F62B", "#008D19", "#004D0D"], # pale / mid / base / dark / black              
                      'oranges':["#FFD6AC", "#FFAC54", "#FF8300", "#B95F00", "#653400"],             
                      'reds':["#FFB2AC", "#FF6154", "#FF1300", "#B90D00", "#650700"],                 
                      'blues':["#A4DCEF", "#54C8EE", "#03B5F0", "#015773", "#012F3F"]}


global OD_WTW, OS_WTW, CAL_VAL_OD, CAL_VAL_OS, EYE_ORING
OD_WTW = 0; 
OS_WTW = 0;
CAL_VAL_OD = 5/33;
CAL_VAL_OS = 5/33;
EYE_ORING = [[0,0],
             [0,0]]

def enclosed_area(xy):
    """ 
        Calculates polygon area.
        x = xy[:,0], y = xy[:,1]
    """
    l = len(xy)
    s = 0.0
    # Python arrys are zer0-based
    for i in range(l):
        j = (i+1)%l  # keep index in [0,l)
        s += (xy[j,0] - xy[i,0])*(xy[j,1] + xy[i,1])
    return -0.5*s
    
def trans_PD(AL,dx,CAL_VAL):
    theta = []
    dx = np.array(dx, dtype=float)
    for i in range(0,dx.size):
        try:
            math.asin((2*abs(dx[i])/AL)*(5/33))
        except:
            dx[i] = np.nan
        if dx[i]<0:
            dx[i] = -dx[i]
            theta = np.append(theta, - 100*math.tan(math.asin((2*dx[i]/AL)*CAL_VAL)))
        else:
            theta = np.append(theta, 100*math.tan(math.asin((2*dx[i]/AL)*CAL_VAL)))
    return np.round(theta,1)

def trans_AG(AL,dx,CAL_VAL):
    theta = []
    dx = np.array(dx, dtype=float)
    for i in range(0,dx.size):
        try:
            Th = math.degrees(math.asin((2*abs(dx[i])/AL)*CAL_VAL))
        except:
            try:
                Th = 90+math.degrees(
                    math.asin(
                        (2*(abs(dx[i])-(AL*330)/(2*50))/AL)*CAL_VAL
                        )
                    )
            except:
                Th = np.nan
        if dx[i]<0:
            dx[i] = -dx[i]
            theta = np.append(theta, - Th)
        else:
            theta = np.append(theta, Th)
    return np.round(theta,1)

def GetVideo(csv_path):
    fall = csv_path.replace(".csv",".avi")
    if not os.path.isfile(fall):
        fall = csv_path.replace(".csv",".mp4") 
    return cv2.VideoCapture(fall)

def DrawEyePosition(frame, eyes, OD_p, OS_p):
    for (ex,ey,ew,eh) in eyes:    
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    if len(np.argwhere(np.isnan(OD_p)))<3:
        cv2.rectangle(frame,
                      (int(OD_p[0]),int(OD_p[1])),
                      (int(OD_p[0])+1,int(OD_p[1])+1),
                      (0,255,0),2)
        cv2.circle(frame,(int(OD_p[0]),int(OD_p[1])),
                   int(OD_p[2]),
                   (255,255,255),2)        
    if len(np.argwhere(np.isnan(OS_p)))<3:
        cv2.rectangle(frame,
                      (int(OS_p[0]),int(OS_p[1])),
                      (int(OS_p[0])+1,int(OS_p[1])+1),
                      (0,255,0),2)
        cv2.circle(frame,(int(OS_p[0]),int(OS_p[1])),
                   int(OS_p[2]),
                   (255,255,255),2)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

class ACT_Save(object):
    _ACT_dx = {'ID':[],
          'Date':[],
          'H_Dx':[],
          'H_Dev':[],
          'H_type':[],
          'V_Dx':[],
          'V_Dev':[],
          }
    
class Gaze9_Save(object):
    _Gaze9_dx = {'ID':[],'Date':[]}
    for i in range(0,len(GAZE_9_TIME)):
        _Gaze9_dx[GAZE_9_TIME[i]+'_OD_H_Dev'] = []
        _Gaze9_dx[GAZE_9_TIME[i]+'_OS_H_Dev'] = []
        _Gaze9_dx[GAZE_9_TIME[i]+'_OD_V_Dev'] = []
        _Gaze9_dx[GAZE_9_TIME[i]+'_OS_V_Dev'] = []
    _Gaze9_dx['OD_Area'] = []
    _Gaze9_dx['OS_Area'] = []
    
class CUT_Save(object):
    _CUT_dx = {'ID':[],
               'Date':[],
          'H_Dx':[],
          'H_Dev':[],
          'H_type':[],
          'V_Dx':[],
          'V_Dev':[],
          }
    
class Neurobit():
    def __init__(self):
        self.version = '2.9.0'
        self.task = str("Subject")
        self.session = []
        self.major_path = os.getcwd()
        self.save_path = os.getcwd()+"\\RESULT\\Version"+self.version
        self.saveVideo_path = []
        self.DB_path = []
        self.CmdTime = []
        self.showVideo = True
        self.AL_OD = 25.15
        self.AL_OS = 25.15
        self.FolderName = []
    def GetFolderPath(self):
# =============================================================================
#         ID, profile, _, _ = self.GetDxSql()
# =============================================================================
        ID, profile, _ = self.GetDxSql()
        f   = open(os.path.join(self.major_path,'ID.ns'), 'r')
        Date = f.readlines()[1].replace('\n','')
        folder      = glob.glob(self.main_path  +
                                "\\Result\\"    + 
                                ID + "\\"       +
                                str(Date+"*"+ ID))[0]
        return folder
    def GetSubjectFiles(self, main_path):          
        self.main_path  = main_path
        return glob.glob(self.GetFolderPath()+"\*OcularMotility.csv")
    def GetDxSql(self):
        f   = open(os.path.join(self.major_path,'ID.ns'), 'r')
        lines = f.readlines()
        ID  = lines[0].replace('\n','')
        Date = lines[1].replace('\n','')
        Date = datetime(int(Date[:4]), int(Date[4:6]), int(Date[6:])).strftime('%Y/%m/%d')
        con = sqlite3.connect(os.path.join(self.major_path,"NeurobitNS01-1.db"))
        cur = con.cursor()
        cur.execute("SELECT * FROM Patient WHERE [ID]='" + ID + "'")
# =============================================================================
#         cur.execute('select * from sqlite_master').fetchall()
#         cur.execute('SELECT * FROM'+ID)
# =============================================================================
        profile = np.array(cur.fetchall())[-1]
        cur.execute("SELECT * FROM Visit WHERE [Patient_ID]='" + ID + "'" + 
                    "AND [Datetime]='"+Date+"'"+
                    "AND [Procedure]='OcularMotility'"
                    "AND [Procedure_ID]='0'")
        visit_ID = np.array(cur.fetchall())[0]
# =============================================================================
#         cur.execute("SELECT * FROM ExamSheet WHERE [Visit_ID]='" + visit_ID[0] + "'")
#         exam_sheet = np.array(cur.fetchall())[0]
#         return ID, profile, exam_sheet, visit_ID
# =============================================================================
        return ID, profile, visit_ID
    def GetProfile(self, csv_path):
        global CAL_VAL_OD, CAL_VAL_OS
        
        cmd_csv = pd.read_csv(csv_path, dtype=object)
# =============================================================================
#         ID, profile, exam_sheet, visit_ID = self.GetDxSql()
# =============================================================================
        ID, profile, visit_ID = self.GetDxSql()        
        
        self.Task   = cmd_csv.Mode[0]        
        self.ID     = cmd_csv.PatientID[0]              
        self.Device = cmd_csv.Device[0]
        self.Doctor = visit_ID[3]
        self.Date   = visit_ID[1].replace("/","")[:8]
        
        tmp = int(np.where(cmd_csv.PatientID == "Eye")[0]+1)
        if self.task == 'ACT':
            self.VoiceCommand = np.array(cmd_csv.PatientID[tmp:], dtype=float)
            #print("GET VoiceCommand")
        elif self.task == '9_Gaze':
            self.VoiceCommand = np.array([cmd_csv.ExaminerID[tmp:],cmd_csv.Device[tmp:],cmd_csv.PatientID[tmp:]], dtype=float)
            #print("GET VoiceCommand")
        elif self.task == 'CUT':
            self.VoiceCommand = np.array(cmd_csv.PatientID[tmp:], dtype=float)
            #print("GET VoiceCommand")
        else:
            pass#print("Go to make "+self.task+" function!!!")
        
        self.Name   = str(profile[2]+","+profile[4])
        self.Gender = str(profile[6])
        self.DoB    = str(profile[7])
        self.Age    = str(int(self.Date[:4])-int(self.DoB.replace("/","")[:4]))  
        self.Height = str(profile[8])
        
# =============================================================================
#         self.Dx         = str(DX[np.where(exam_sheet[3:10]=='True')[0]]) + ", " + exam_sheet[-5] + ", " + visit_ID[-1]
#         self.VA_OD      = str(exam_sheet[11])
#         self.BCVA_OD    = str(exam_sheet[12])
#         self.Ref_OD     = str(exam_sheet[13])
#         self.pupil_OD   = str(exam_sheet[14])
#         try: 
#             self.WTW_OD     = float(exam_sheet[15]) 
#             CAL_VAL_OD      = self.WTW_OD/OD_WTW            
#         except: self.WTW_OD  = np.nan
#         try: self.AL_OD      = float(exam_sheet[16])
#         except: self.AL_OD  = np.nan
#         
#         self.VA_OS      = str(exam_sheet[17])
#         self.BCVA_OS    = str(exam_sheet[18])
#         self.Ref_OS     = str(exam_sheet[19])
#         self.pupil_OS   = str(exam_sheet[20])          
#         try: 
#             self.WTW_OS     = float(exam_sheet[21])
#             CAL_VAL_OS      = self.WTW_OS/OS_WTW            
#         except: self.WTW_OS  = np.nan
#         try: self.AL_OS      = float(exam_sheet[22])
#         except: self.AL_OS  = np.nan
#         
#         self.PD         = str(exam_sheet[23])            
#         self.Hertal_OD  = str(exam_sheet[24])
#         self.Hertal_OS  = str(exam_sheet[25])
#         self.Hertal_Len = str(exam_sheet[26])
#         self.Stereo     = str(exam_sheet[27])
# =============================================================================
                            
    def GetEyePosition(self):
        cap = GetVideo(self.csv_path)
        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]
        OD_x = EYE_ORING[0][0]
        OD_y = EYE_ORING[0][1]
        OS_x = EYE_ORING[1][0]
        OS_y = EYE_ORING[1][1]
        print(OD_x, OD_y, OS_x, OS_y)
        if(OD_x-150>0):
            eyes = [[int(OD_x-175),int(OD_y-125),350,250],
                           [int(OS_x-175),int(OS_y-125),350,250]]
        else:
            eyes = [[0,int(OD_y-125),OD_x*2,250],
                           [int(OS_x-175),int(OS_y-125),350,250]]
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        out = cv2.VideoWriter(os.path.join(self.saveVideo_path,self.FileName+'.mp4'),
                          fourcc, 25, (width,height))
        #eyes, OD_pre, OS_pre = get_eye_position(GetVideo(self.csv_path),eyes_origin)
        OD = []; OS = []; thr_eyes = [] 
        frame_cnt = 0; OD_cal_cnt = 0; OS_cal_cnt = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap = GetVideo(self.csv_path)
        while(cap.isOpened()):
            cap.set(1,frame_cnt)
            ret, frame = cap.read()
            if ret == True:
                frame_cnt+=1                             
                OD_p,OS_p,thr = capture_eye_pupil(frame,eyes)
                if (not np.isnan(OD_p).any() and 
                    eyes[0][0]<OD_p[0]<eyes[0][0]+eyes[0][2] and 
                    eyes[0][1]<OD_p[1]<eyes[0][1]+eyes[0][3]):
                    OD.append([int(OD_p[0]),int(OD_p[1]), int(OD_p[2])])
                else:
                    OD.append([np.nan,np.nan,np.nan])
                    #print("An OD exception occurred")
                if (not np.isnan(OS_p).any() and 
                    eyes[1][0]<OS_p[0]<eyes[1][0]+eyes[1][2] and 
                    eyes[1][1]<OS_p[1]<eyes[1][1]+eyes[1][3]):
                    OS.append([int(OS_p[0]), int(OS_p[1]), int(OS_p[2])])
                else:
                    OS.append([np.nan,np.nan,np.nan])
                    #print("An OS exception occurred")                
                DrawEyePosition(frame, eyes, OD[-1], OS[-1])
                self.DrawTextVideo(frame, frame_cnt)
                
                for (ex,ey,ew,eh) in eyes:    
                    cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                
                if self.showVideo:
                    cv2.imshow('frame',frame) 
                    cv2.waitKey(1)  
                
                out.write(frame)
                
            else:
                break    
            time.sleep(0.001)
            pbar.update(1)
        if self.showVideo:            
            cv2.destroyAllWindows()
        out.release()
        
        self.OD = np.array(OD).transpose()
        self.OS = np.array(OS).transpose()
        self.Preprocessing()
    def MergeFile(self):
        if len(self.session)>1:
            csv_1 = pd.read_csv(self.session[0], dtype=object)
            videoList = []
            videoList.append(VideoFileClip(self.session[0].replace(".csv",".mp4")))
            for i in range(1,len(self.session)):                
                csv_2 = pd.read_csv(self.session[i], dtype=object)
                tmp = int(np.where(csv_2.PatientID == "Eye")[0]+1)
                csv_1 = csv_1.append(csv_2[tmp:], ignore_index=True)
                video = VideoFileClip(self.session[i].replace(".csv",".mp4"))
                videoList.append(video)
                              
            final_video = concatenate_videoclips(videoList)
            final_video.write_videofile(os.path.join(self.saveMerge_path,self.FolderName + "_" + self.task + ".mp4"))  
            csv_1.to_csv(os.path.join(self.saveMerge_path,self.FolderName + "_" + self.task + ".csv"))        
            self.csv_path = os.path.join(self.saveMerge_path,self.FolderName + "_" + self.task + ".csv")
            self.FileName = self.csv_path.split('\\')[-1].replace(".csv","")
        else:
            self.csv_path = self.session[0]
    def GetCommand(self):    
        self.GetProfile(self.csv_path)
        self.GetTimeFromCmd()
        self.IsVoiceCommand = True
    def Save2Cloud(self):
        gauth = GoogleAuth()       
        drive = GoogleDrive(gauth) 
        upload_file = self.FileName+".mp4"
       	gfile = drive.CreateFile({'parents': [{'id': '14MSiEu6cHcsXAElmV9O7vIQj8rlNVCHs'}]})
       	# Read file and set it as the content of this instance.
        os.chdir(self.saveVideo_path)
       	gfile.SetContentFile(upload_file)
        os.chdir(self.major_path)
       	gfile.Upload() # Upload the file. 
        # Check update or not
        NotUpdated = True
        while NotUpdated:
            file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('14MSiEu6cHcsXAElmV9O7vIQj8rlNVCHs')}).GetList()
            for file in file_list:
                if file['title'] == self.FileName+".mp4":
                    NotUpdated = False        
    def DrawQRCode(self):
        os.chdir(self.major_path)
        gauth = GoogleAuth()       
        drive = GoogleDrive(gauth) 
       	# Read file and set it as the content of this instance.
        file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('14MSiEu6cHcsXAElmV9O7vIQj8rlNVCHs')}).GetList()
        for file in file_list:
            if file['title'] == self.FileName+".mp4":
                self.website =file['alternateLink']
        img = qrcode.make(self.website)
        img.save(os.path.join(self.saveImage_path,"QR_code.png"))
    def Preprocessing(self):
        win = 15        # window         
        # median moving window
        OD_temp = self.OD; OS_temp = self.OS;
        for i in range(win,(len(self.OD[0,:])-win)):
            if not np.isnan(self.OD[0,i]): OD_temp[:,i] = np.nanmedian(self.OD[:,i-win:i+win],axis = 1)
            if not np.isnan(self.OS[0,i]): OS_temp[:,i] = np.nanmedian(self.OS[:,i-win:i+win],axis = 1)        
        self.OD = OD_temp
        self.OS = OS_temp
        
        