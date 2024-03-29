# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 23:46:56 2021

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
import shutil
import time
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from PIL import Image
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip, concatenate_videoclips
from function_eye_capture import capture_eye_iris, capture_eye_pupil, get_eye_position
from datetime import datetime
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# google cloud function
from google.cloud import storage

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
GAZE_9_STR      = ["Down",       "Front",    "Left",     
                   "Left Down",  "Left Up",  "Right",  
                   "Right Down", "Right Up", "Up"]
GAZE_9_EYEFIG = [5,2,8,4,6,1,3,7,9]

# color map
line_color_palatte = {'greens':["#A5F5B3", "#51F46D",   "#00F62B", "#008D19", "#004D0D"], # pale / mid / base / dark / black              
                      'oranges':["#FFD6AC", "#FFAC54", "#FF8300", "#B95F00", "#653400"],             
                      'reds':["#FFB2AC", "#FF6154", "#FF1300", "#B90D00", "#650700"],                 
                      'blues':["#A4DCEF", "#54C8EE", "#03B5F0", "#015773", "#012F3F"]}

global OD_WTW, OS_WTW, CAL_VAL_OD, CAL_VAL_OS
OD_WTW = 0; 
OS_WTW = 0;
CAL_VAL_OD = 5/33;
CAL_VAL_OS = 5/33;

def enclosed_area(x_lower, y_lower, x_upper, y_upper):
    return(np.trapz(y_upper, x=x_upper) - np.trapz(y_lower, x=x_lower))
    
def trans_PD(AL,dx,CAL_VAL):
    theta = []
    dx = np.array(dx)
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
    dx = np.array(dx)
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

class Neurobit():
    def __init__(self):
        self.version = '2.7.0' 
        self.major_path = os.getcwd()
        self.task = str("Subject")
        self.session = []
        self.saveVideo_path = []
        self.CmdTime = []
        self.showVideo = True
        self.AL_OD = 25.15
        self.AL_OS = 25.15
        self.dx_9Gaze = { 'ID':[],
                          'Examine Date':[],#'T_degree_OD':[],'T_degree_OS':[],
                          'X_D':[],'Y_D':[],
                          'X_F':[],'Y_F':[],
                          'X_L':[],'Y_L':[],
                          'X_LD':[],'Y_LD':[],
                          'X_LU':[],'Y_LU':[], 
                          'X_R':[],'Y_R':[],
                          'X_RD':[],'Y_RD':[],
                          'X_RU':[],'Y_RU':[],
                          'X_U':[],'Y_U':[]}
        self.dx_ACT = {'ID':[],     'Examine Date':[],  'Orthophoria':[],
                       'XT':[],     'XT type':[],       'ET':[],            'ET type':[],
                       'LHT':[],    'RHT':[],           'LHoT':[],          'RHoT':[]}
    def GetFolderPath(self):
        ID, profile, _, _ = self.GetDxSql()
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
        cur.execute("SELECT * FROM ExamSheet WHERE [Visit_ID]='" + visit_ID[0] + "'")
        exam_sheet = np.array(cur.fetchall())[0]
        return ID, profile, exam_sheet, visit_ID
    def GetProfile(self, csv_path):
        global CAL_VAL_OD, CAL_VAL_OS
        
        cmd_csv = pd.read_csv(csv_path, dtype=object)
        ID, profile, exam_sheet, visit_ID = self.GetDxSql()        
        
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
        
        self.Dx         = str(DX[np.where(exam_sheet[3:10]=='True')[0]]) + ", " + exam_sheet[-5] + ", " + visit_ID[-1]
        self.VA_OD      = str(exam_sheet[11])
        self.BCVA_OD    = str(exam_sheet[12])
        self.Ref_OD     = str(exam_sheet[13])
        self.pupil_OD   = str(exam_sheet[14])
        try: 
            self.WTW_OD     = float(exam_sheet[15]) 
            CAL_VAL_OD      = self.WTW_OD/OD_WTW            
        except: self.WTW_OD  = np.nan
        try: self.AL_OD      = float(exam_sheet[16])
        except: self.AL_OD  = np.nan
        
        self.VA_OS      = str(exam_sheet[17])
        self.BCVA_OS    = str(exam_sheet[18])
        self.Ref_OS     = str(exam_sheet[19])
        self.pupil_OS   = str(exam_sheet[20])          
        try: 
            self.WTW_OS     = float(exam_sheet[21])
            CAL_VAL_OS      = self.WTW_OS/OS_WTW            
        except: self.WTW_OS  = np.nan
        try: self.AL_OS      = float(exam_sheet[22])
        except: self.AL_OS  = np.nan
        
        self.PD         = str(exam_sheet[23])            
        self.Hertal_OD  = str(exam_sheet[24])
        self.Hertal_OS  = str(exam_sheet[25])
        self.Hertal_Len = str(exam_sheet[26])
        self.Stereo     = str(exam_sheet[27])
                            
    def GetEyePosition(self):
        cap = GetVideo(self.csv_path)
        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        eyes_origin = [[int(width/4-200),int(height/2)-100,375,200],
                       [int(width/2+75),int(height/2)-100,375,200]]
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        out = cv2.VideoWriter(os.path.join(self.saveVideo_path,self.FileName+'.mp4'),
                          fourcc, 25, (width,height))
        eyes, OD_pre, OS_pre = get_eye_position(GetVideo(self.csv_path),eyes_origin)
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
                    eyes_origin[0][0]<OD_p[0]<eyes_origin[0][0]+eyes_origin[0][2] and 
                    eyes_origin[0][1]<OD_p[1]<eyes_origin[0][1]+eyes_origin[0][3]):
                    OD.append([int(OD_p[0]),int(OD_p[1]), int(OD_p[2])])
                else:
                    OD.append([np.nan,np.nan,np.nan])
                    #print("An OD exception occurred")
                if (not np.isnan(OS_p).any() and 
                    eyes_origin[1][0]<OS_p[0]<eyes_origin[1][0]+eyes_origin[1][2] and 
                    eyes_origin[1][1]<OS_p[1]<eyes_origin[1][1]+eyes_origin[1][3]):
                    OS.append([int(OS_p[0]), int(OS_p[1]), int(OS_p[2])])
                else:
                    OS.append([np.nan,np.nan,np.nan])
                    #print("An OS exception occurred")                
                DrawEyePosition(frame, eyes, OD[-1], OS[-1])
                self.DrawTextVideo(frame, frame_cnt)
                
                for (ex,ey,ew,eh) in eyes_origin:    
                    cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                
                if self.showVideo:
                    cv2.imshow('frame',frame) 
                    cv2.waitKey(1)  
                
                out.write(frame)
                dOD = np.sum(np.abs(np.array(OD[-1])-OD_pre))
                dOS = np.sum(np.abs(np.array(OS[-1])-OS_pre))
                if np.logical_or(dOD>60, np.isnan(dOD)) and OD_cal_cnt <= 0:
                    OD_cal_cnt = 60
                    eyes, OD_pre, OS_pre = get_eye_position(cap,eyes_origin)    
                elif np.logical_or(dOS>60, np.isnan(dOS)) and OD_cal_cnt <= 0 and OS_cal_cnt <= 0:
                    OS_cal_cnt = 60
                    eyes, OD_pre, OS_pre = get_eye_position(cap,eyes_origin) 
                OD_cal_cnt-=1; OS_cal_cnt-=1
                
            else:
                break    
            time.sleep(0.001)
            pbar.update(1)
        if self.showVideo:            
            cv2.destroyAllWindows()
        out.release()
        self.OD = np.array(OD).transpose()
        self.OS = np.array(OS).transpose()
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
    
class ACT_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "ACT"
        self.sequence = 0
        self.FolderName = csv_path.split('\\')[-2]
        self.FileName = csv_path.split('\\')[-1].replace(".csv","")
        self.main_path = csv_path.replace("\\"+csv_path.split('\\')[-2],"").replace("\\"+csv_path.split('\\')[-1],"")
        self.save_MainPath = self.main_path+"\\"+self.FolderName
        self.saveReport_path = self.save_MainPath
        self.saveMerge_path = self.save_MainPath+"\\"+self.task
        self.saveVideo_path = self.save_MainPath+"\\"+self.task+"\\HoughCircle"
        self.saveImage_path = self.save_MainPath+"\\"+self.task+"\\Image"              
        if not os.path.isdir(self.saveVideo_path):
            os.makedirs(self.saveVideo_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
    def Exec(self):
        self.GetCommand()        
        self.GetEyePosition()  
        self.SeperateSession()              
        self.FeatureExtraction()  
        self.GetDiagnosis()  
        self.Save2Cloud()
        
        self.DrawEyeFig()
        self.DrawEyeTrack()  
        self.DrawQRCode()
    def FeatureExtraction(self):
        OD = self.OD; OS = self.OS
        OD_ACT = []; OS_ACT = [];       # all position in each ACT_TIME
        for i in range(0,len(ACT_TIME)):
            temp = self.CmdTime[ACT_TIME[i]]
# =============================================================================
#             delete = np.where(temp>len(OD[0])-1)[0]
#             if delete.any():
#                 temp = np.delete(temp, delete)
# =============================================================================
            if type(self.CmdTime[ACT_TIME[i]]) == list:
                OD_ACT_tmp = []; OS_ACT_tmp = [];
                thr_odx = 50; thr_ody = 50; thr_osx = 50; thr_osy = 50
                for temp in self.CmdTime[ACT_TIME[i]]:
                    diff_x = np.diff(OD[0][temp])
                    grad_x = np.sign(diff_x)
                    plateau_x = OD[0][temp][np.where(grad_x==0)[0]+1]
                    
                    diff_y = np.diff(OD[1][temp])
                    grad_y = np.sign(diff_y)
                    plateau_y = OD[1][temp][np.where(grad_y==0)[0]+1]
                    
                    diff_p = np.diff(OD[2][temp])
                    grad_p = np.sign(diff_p)
                    plateau_p = OD[2][temp][np.where(grad_p==0)[0]+1]
                    
                    OD_ACT_tmp.append(np.round(
                        [np.nanpercentile(plateau_x, 50),     # x axis
                         np.nanpercentile(plateau_y, 50),     # y axis
                         np.nanpercentile(plateau_p, 50)],2)) # pupil size
                    
                    diff_x = np.diff(OS[0][temp])
                    grad_x = np.sign(diff_x)
                    plateau_x = OS[0][temp][np.where(grad_x==0)[0]+1]
                    
                    diff_y = np.diff(OS[1][temp])
                    grad_y = np.sign(diff_y)
                    plateau_y = OS[1][temp][np.where(grad_y==0)[0]+1]
                    
                    diff_p = np.diff(OS[2][temp])
                    grad_p = np.sign(diff_p)
                    plateau_p = OS[2][temp][np.where(grad_p==0)[0]+1]
                    
                    OS_ACT_tmp.append(np.round(
                        [np.nanpercentile(plateau_x, 50),     # x axis
                         np.nanpercentile(plateau_y, 50),     # y axis
                         np.nanpercentile(plateau_p, 50)],2)) # pupil size
                    
                OD_ACT_tmp = np.array(OD_ACT_tmp)
                OS_ACT_tmp = np.array(OS_ACT_tmp)
                OD_ACT.append(np.round(
                    [np.nanpercentile(OD_ACT_tmp[:,0], 50),     # x axis
                     np.nanpercentile(OD_ACT_tmp[:,1], 50),     # y axis
                     np.nanpercentile(OD_ACT_tmp[:,2], 50)],2))
                OS_ACT.append(np.round(
                    [np.nanpercentile(OS_ACT_tmp[:,0], 50),     # x axis
                     np.nanpercentile(OS_ACT_tmp[:,1], 50),     # y axis
                     np.nanpercentile(OS_ACT_tmp[:,2], 50)],2))
            else:
                diff_x = np.diff(OD[0][temp])
                grad_x = np.sign(diff_x)
                plateau_x = OD[0][temp][np.where(grad_x==0)[0]+1]
                
                diff_y = np.diff(OD[1][temp])
                grad_y = np.sign(diff_y)
                plateau_y = OD[1][temp][np.where(grad_y==0)[0]+1]
                
                diff_p = np.diff(OD[2][temp])
                grad_p = np.sign(diff_p)
                plateau_p = OD[2][temp][np.where(grad_p==0)[0]+1]
                
                OD_ACT.append(np.round(
                    [np.nanpercentile(plateau_x, 50),     # x axis
                     np.nanpercentile(plateau_y, 50),     # y axis
                     np.nanpercentile(plateau_p, 50)],2)) # pupil size
                diff_x = np.diff(OS[0][temp])
                grad_x = np.sign(diff_x)
                plateau_x = OS[0][temp][np.where(grad_x==0)[0]+1]
                
                diff_y = np.diff(OS[1][temp])
                grad_y = np.sign(diff_y)
                plateau_y = OS[1][temp][np.where(grad_y==0)[0]+1]
                
                diff_p = np.diff(OS[2][temp])
                grad_p = np.sign(diff_p)
                plateau_p = OS[2][temp][np.where(grad_p==0)[0]+1]
                
                OS_ACT.append(np.round(
                    [np.nanpercentile(plateau_x, 50),     # x axis
                     np.nanpercentile(plateau_y, 50),     # y axis
                     np.nanpercentile(plateau_p, 50)],2)) # pupil size
        
        # ET、XT angle
        OD_ACT = np.array(OD_ACT)
        OS_ACT = np.array(OS_ACT)
        self.OD_ACT = OD_ACT
        self.OS_ACT = OS_ACT
        # Fixation_eye - Covered_eye
        OD_fix = OD_ACT[1]-OD_ACT[2]    # CL-CR
        OS_fix = OS_ACT[2]-OS_ACT[1]    # CR-CL
        try:
            OD_fix = np.append(trans_PD(self.AL_OD,OD_fix[0:2]), OD_fix[2],CAL_VAL_OD)
            OS_fix = np.append(trans_PD(self.AL_OS,OS_fix[0:2]), OS_fix[2],CAL_VAL_OS)
        except:
            pass#print("No profile")
        self.OD_fix = OD_fix        # one position in each ACT_TIME
        self.OS_fix = OS_fix
    def GetDiagnosis(self):
        OD_fix = self.OD_fix; OS_fix = self.OS_fix
        thr =1.5
        self.NeurobitDx_H = None
        self.NeurobitDx_V = None
        self.NeurobitDxTp_X = None
        if np.all(np.abs([OD_fix,OS_fix])<=thr):
            self.Ortho = True
            self.NeurobitDx_H = 'Ortho'
            self.NeurobitDx_V = 'Ortho'
            self.NeurobitDxTp_H = 'None'
            self.NeurobitDxDev_H = 0
            self.NeurobitDxDev_V = 0
        else:
            self.Ortho = False
        
        if -OS_fix[0]>thr or OD_fix[0]>thr:
            self.NeurobitDx_H = 'XT'
            if -OS_fix[0]>thr and OD_fix[0]>thr:
                self.NeurobitDxTp_H = 'Divergence'
                self.NeurobitDxDev_H = (abs(OD_fix[0])+abs(OS_fix[0]))/2
            elif -OS_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OS_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OS, Levoversion'
                else:
                    self.NeurobitDxTp_X = 'OS, Divergence'
            elif OD_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OD_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OD, Levoversion'
                else:
                    self.NeurobitDxTp_X = 'OD, Divergence'
        elif OS_fix[0]>thr or -OD_fix[0]>thr:
            self.NeurobitDx_H = 'ET'
            if OS_fix[0]>thr and -OD_fix[0]>thr:
                self.NeurobitDxTp_H = 'Convergence'
                self.NeurobitDxDev_H = (abs(OD_fix[0])+abs(OS_fix[0]))/2
            elif OS_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OS_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OS Detroversion'
                else:
                    self.NeurobitDxTp_X = 'OS Convergence'           
            elif -OD_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OD_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OD Detroversion'
                else:
                    self.NeurobitDxTp_X = 'OD Convergence'
        else:
            self.NeurobitDx_H = 'Ortho'
            self.NeurobitDxDev_H = 0
        
        if OS_fix[1]>thr or -OD_fix[1]>thr:
            self.NeurobitDx_V = 'LHT'
            if OS_fix[1]>thr:
                self.NeurobitDxDev_V = abs(OS_fix[1])
            else:
                self.NeurobitDxDev_V = abs(OD_fix[1])
        
        elif -OS_fix[1]>thr or OD_fix[1]>thr:
            self.NeurobitDx_V = 'LHoT'
            if OS_fix[1]>thr:
                self.NeurobitDxDev_V = abs(OS_fix[1])
            else:
                self.NeurobitDxDev_V = abs(OD_fix[1])
        else:
            self.NeurobitDx_V = 'Ortho'
            self.NeurobitDxDev_V = 0            
    def GetTimeFromCmd(self):
        cmd = self.VoiceCommand
        O_t = np.where(cmd==0)[0]
        CL_t = np.where(np.logical_or(cmd==1,cmd==3))[0]
        CR_t = np.where(cmd==2)[0] 
        UCL_t = np.where(cmd==4)[0]
        self.CmdTime = {"CL_t": np.array(CL_t),
                        "CR_t": np.array(CR_t),
                        "O_t":  np.array(O_t),
                        "UCR_t":np.array(UCL_t)}
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(ACT_TIME)):
            temp = np.array(self.CmdTime[ACT_TIME[i]])
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)
            diff_temp = np.diff(temp)
            inds = np.where(diff_temp>20)[0]
            if len(inds)>0:
                list_temp = list(); j = 0
                for ind in inds:
                    list_temp.append(temp[j:ind])
                    j = ind
                list_temp.append(temp[ind:])
                self.CmdTime[ACT_TIME[i]] = list_temp
            else:
                self.CmdTime[ACT_TIME[i]] = temp
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/25
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(EYE)):
            if EYE[i] == 'OD':
                x_diff = self.OD_ACT[0,0]-OD[0,:]
                y_diff = self.OD_ACT[0,1]-OD[1,:]
                x_PD = trans_PD(self.AL_OD,x_diff,CAL_VAL_OD)
                y_PD = trans_PD(self.AL_OD,y_diff,CAL_VAL_OD)
            else:
                x_diff = self.OS_ACT[0,0]-OS[0,:]
                y_diff = self.OS_ACT[0,1]-OS[1,:]
                x_PD = trans_PD(self.AL_OS,x_diff,CAL_VAL_OS)
                y_PD = trans_PD(self.AL_OS,y_diff,CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position (PD)")
            plt.title("Alternated Cover Test "+ EYE[i])
            
            plt.grid(True, linestyle=':')
            plt.xticks(fontsize= 8)
            plt.yticks(fontsize= 8)
            
            plt.text(0,90, "right",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(0,-90, "left",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1], 90,"up",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1], -90,"down",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8) 
            plt.ylim([-100,100])
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300) 
    def DrawEyeFig(self):
        ACT = []; OD = self.OD; OS = self.OS
        for i in range(0,len(self.OS_ACT)):
            try:t = np.concatenate(np.array(self.CmdTime[ACT_TIME[i]]))
            except:t = self.CmdTime[ACT_TIME[i]]
            if not np.isnan(self.OD_ACT[i,0]) and not np.isnan(self.OS_ACT[i,0]):
                OD_diff = abs(OD[0,t]-self.OD_ACT[i,0])+abs(OD[1,t]-self.OD_ACT[i,1])
                OS_diff = abs(OS[0,t]-self.OS_ACT[i,0])+abs(OS[1,t]-self.OS_ACT[i,1])
                Diff = np.sum(np.array([OD_diff, OS_diff]),axis = 0)
                pupil = OS[2,t]+OD[2,t]
            elif np.isnan(self.OD_ACT[i,0]):
                Diff = abs(OS[0,t]-self.OS_ACT[i,0])+abs(OS[1,t]-self.OS_ACT[i,1])
                pupil = OS[2,t]
            else:
                Diff = abs(OD[0,t]-self.OD_ACT[i,0])+abs(OD[1,t]-self.OD_ACT[i,1])
                pupil = OD[2,t]
            try:
                #ind = np.where(Diff == np.nanmin(Diff))[0]
                #ind_pu = np.where(pupil[ind] == np.nanmax(pupil[ind]))[0]
                ACT.append(t[np.where(Diff == np.nanmin(Diff))[0][0]])
            except:
                ACT.append(ACT[-1])
                #print("Not Detect "+ ACT_TIME[i]) 
        pic_cont = 1
        empt=0
        #fig = plt.figure(figsize=(11.7,8.3))
        fig = plt.gcf()
        fig.set_size_inches(3,4.4, forward=True)
        fig.set_dpi(300)
        for pic in ACT:
            cap = GetVideo(self.csv_path)
            cap.set(1,pic)
            ret, im = cap.read()
            height = im.shape[0]
            width = im.shape[1]
            try:
                cv2.rectangle(im,
                              (int(self.OD_ACT[pic_cont-1][0]),int(self.OD_ACT[pic_cont-1][1])),
                              (int(self.OD_ACT[pic_cont-1][0])+1,int(self.OD_ACT[pic_cont-1][1])+1),
                              (0,255,0),2)
                cv2.circle(im,(int(self.OD_ACT[pic_cont-1][0]),int(self.OD_ACT[pic_cont-1][1])),
                           int(self.OD_ACT[pic_cont-1][2]),
                           (255,255,255),2) 
            except:
                pass#print("OD Absent!")
            try:
                cv2.rectangle(im,
                              (int(self.OS_ACT[pic_cont-1][0]),int(self.OS_ACT[pic_cont-1][1])),
                              (int(self.OS_ACT[pic_cont-1][0])+1,int(self.OS_ACT[pic_cont-1][1])+1),
                              (0,255,0),2)
                cv2.circle(im,(int(self.OS_ACT[pic_cont-1][0]),int(self.OS_ACT[pic_cont-1][1])),
                           int(self.OS_ACT[pic_cont-1][2]),
                           (255,255,255),2)
            except:
                pass#print("OS Absent!")
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _,thresh_1 = cv2.threshold(gray,110,255,cv2.THRESH_TRUNC)
            exec('ax'+str(pic_cont)+'=plt.subplot(4, 1, pic_cont)')
            exec('ax'+str(pic_cont)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
            exec('ax'+str(pic_cont)+'.axes.xaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.axes.yaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.set_ylim(int(3*height/4),int(height/4))')
# =============================================================================
#             exec('ax'+str(pic_cont)+ '.set_ylim(int(height),int(0))')
# =============================================================================
            exec('ax'+str(pic_cont)+ '.set_ylabel(ACT_LABEL[pic_cont-1])')
            plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]
        for i in range(0,len(ACT_TIME)):
            if frame_cnt in self.CmdTime[ACT_TIME[i]]:
                text = ACT_STR[i]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)[0]
                textX = int((width - textsize[0]) / 2)
                cv2.putText(frame,text, (textX, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 
                            2, (255, 255, 255),
                            2, cv2.LINE_AA)
        if self.IsVoiceCommand:
            text = "Voice Command"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)[0]
            textX = int((width - textsize[0]) / 2)
            cv2.putText(frame,text, (textX, 550), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (0, 255, 255),
                        1, cv2.LINE_AA)
        else:
            text = "No Voice Command"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)[0]
            textX = int((width - textsize[0]) / 2)
            cv2.putText(frame,text, (textX, 550), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (0, 255, 255),
                        1, cv2.LINE_AA)
    
            
class Gaze9_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "9_Gaze"
        self.FolderName = csv_path.split('\\')[-2]
        self.FileName = csv_path.split('\\')[-1].replace(".csv","")
        self.main_path = csv_path.replace("\\"+csv_path.split('\\')[-2],"").replace("\\"+csv_path.split('\\')[-1],"")
        self.save_MainPath = self.main_path+"\\"+self.FolderName
        self.saveReport_path = self.save_MainPath
        self.saveMerge_path = self.save_MainPath+"\\"+self.task
        self.saveVideo_path = self.save_MainPath+"\\"+self.task+"\\HoughCircle"
        self.saveImage_path = self.save_MainPath+"\\"+self.task+"\\Image"              
        if not os.path.isdir(self.saveVideo_path):
            os.makedirs(self.saveVideo_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
    def Exec(self,*args):
        Finished_ACT = False
        for arg in args:
            if arg: 
                Finished_ACT = True
                ACT_Task = arg
        self.GetCommand()        
        self.GetEyePosition()
        self.SeperateSession()                
        if Finished_ACT: self.FeatureExtraction(ACT_Task) 
        else: self.FeatureExtraction() 
        self.Save2Cloud()
               
        self.DrawEyeFig()
        self.DrawEyeMesh()
        self.DrawEyeTrack()  
        self.DrawQRCode() 
    def GetTimeFromCmd(self):    
        #if int(self.Date) >= 20210601:
        #    stupid_bug = np.where(self.VoiceCommand[0,:] == self.VoiceCommand[0,0])[0]
        #    self.VoiceCommand[:,stupid_bug] = 0
    
        x = np.round(self.VoiceCommand[0,:],2); y = np.round(self.VoiceCommand[1,:],2); self.CmdTime = dict()
        cover = self.VoiceCommand[2,:]
        
        x_q1 = np.unique(x)[0]
        x_q2 = np.unique(x)[1]
        x_q3 = np.unique(x)[-1]

        y_q1 = np.unique(y)[0]
        y_q2 = np.unique(y)[1]
        y_q3 = np.unique(y)[-1]
        
        D = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), y == y_q1))[0]#[60:]
        F = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), np.logical_and(y < y_q3, y > y_q1)))[0]#[60:]
        L = np.where(np.logical_and(np.logical_and(x == x_q1, y < y_q3), y > y_q1))[0]#[60:]
        LD = np.where(np.logical_and(x == x_q1, y == y_q1))[0]#[60:]
        LU = np.where(np.logical_and(x == x_q1, y == y_q3))[0]#[60:]
        R = np.where(np.logical_and(np.logical_and(x == x_q3 ,  y < y_q3), y > y_q1))[0]#[60:]
        RD = np.where(np.logical_and(x == x_q3, y == y_q1))[0]#[60:]
        RU = np.where(np.logical_and(x == x_q3, y == y_q3))[0]#[60:]
        U = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), y == y_q3))[0]#[60:]
        
        for i in range(0,len(GAZE_9_TIME)):
            exec('cov_non = np.where(cover['+GAZE_9_TIME[i]+'] == 0)[0]')
            exec('self.CmdTime[GAZE_9_TIME[i]] = '+GAZE_9_TIME[i]+'[cov_non]')
    def FeatureExtraction(self,*args):
        OD = self.OD; OS = self.OS; Finished_ACT = False
        Gaze_9_OD = []; Gaze_9_OS = [];       # all position in each ACT_TIME
        NeurobitDxDev_H = list();NeurobitDxDev_V = list()
        for i in range(0,len(GAZE_9_TIME)):
            temp = self.CmdTime[GAZE_9_TIME[i]]
# =============================================================================
#             delete = np.where(temp>len(OD[0])-1)[0]
#             if delete.any():
#                 temp = np.delete(temp, delete)
# =============================================================================
            if type(self.CmdTime[GAZE_9_TIME[i]]) == list:
                OD_ACT_tmp = []; OS_ACT_tmp = [];
                for temp in self.CmdTime[GAZE_9_TIME[i]]:
                    diff_x = np.diff(OD[0][temp])
                    grad_x = np.sign(diff_x)
                    plateau_x = OD[0][temp][np.where(grad_x==0)[0]+1]
                    
                    diff_y = np.diff(OD[1][temp])
                    grad_y = np.sign(diff_y)
                    plateau_y = OD[1][temp][np.where(grad_y==0)[0]+1]
                    
                    diff_p = np.diff(OD[2][temp])
                    grad_p = np.sign(diff_p)
                    plateau_p = OD[2][temp][np.where(grad_p==0)[0]+1]
                    
                    OD_ACT_tmp.append(np.round(
                        [np.nanpercentile(plateau_x, 50),     # x axis
                         np.nanpercentile(plateau_y, 50),     # y axis
                         np.nanpercentile(plateau_p, 50)],2)) # pupil size
                    
                    diff_x = np.diff(OS[0][temp])
                    grad_x = np.sign(diff_x)
                    plateau_x = OS[0][temp][np.where(grad_x==0)[0]+1]
                    
                    diff_y = np.diff(OS[1][temp])
                    grad_y = np.sign(diff_y)
                    plateau_y = OS[1][temp][np.where(grad_y==0)[0]+1]
                    
                    diff_p = np.diff(OS[2][temp])
                    grad_p = np.sign(diff_p)
                    plateau_p = OS[2][temp][np.where(grad_p==0)[0]+1]
                    
                    OS_ACT_tmp.append(np.round(
                        [np.nanpercentile(plateau_x, 50),     # x axis
                         np.nanpercentile(plateau_y, 50),     # y axis
                         np.nanpercentile(plateau_p, 50)],2)) # pupil size
                    
                OD_ACT_tmp = np.array(OD_ACT_tmp)
                OS_ACT_tmp = np.array(OS_ACT_tmp)
                Gaze_9_OD.append(np.round(
                    [np.nanpercentile(OD_ACT_tmp[:,0], 50),     # x axis
                     np.nanpercentile(OD_ACT_tmp[:,1], 50),     # y axis
                     np.nanpercentile(OD_ACT_tmp[:,2], 50)],2))
                Gaze_9_OS.append(np.round(
                    [np.nanpercentile(OS_ACT_tmp[:,0], 50),     # x axis
                     np.nanpercentile(OS_ACT_tmp[:,1], 50),     # y axis
                     np.nanpercentile(OS_ACT_tmp[:,2], 50)],2))
            else:
                diff_x = np.diff(OD[0][temp])
                grad_x = np.sign(diff_x)
                plateau_x = OD[0][temp][np.where(grad_x==0)[0]+1]
                
                diff_y = np.diff(OD[1][temp])
                grad_y = np.sign(diff_y)
                plateau_y = OD[1][temp][np.where(grad_y==0)[0]+1]
                
                diff_p = np.diff(OD[2][temp])
                grad_p = np.sign(diff_p)
                plateau_p = OD[2][temp][np.where(grad_p==0)[0]+1]
                
                Gaze_9_OD.append(np.round(
                    [np.nanpercentile(plateau_x, 50),     # x axis
                     np.nanpercentile(plateau_y, 50),     # y axis
                     np.nanpercentile(plateau_p, 50)],2)) # pupil size
                
                diff_x = np.diff(OS[0][temp])
                grad_x = np.sign(diff_x)
                plateau_x = OS[0][temp][np.where(grad_x==0)[0]+1]
                
                diff_y = np.diff(OS[1][temp])
                grad_y = np.sign(diff_y)
                plateau_y = OS[1][temp][np.where(grad_y==0)[0]+1]
                
                diff_p = np.diff(OS[2][temp])
                grad_p = np.sign(diff_p)
                plateau_p = OS[2][temp][np.where(grad_p==0)[0]+1]
                
                Gaze_9_OS.append(np.round(
                    [np.nanpercentile(plateau_x, 50),     # x axis
                     np.nanpercentile(plateau_y, 50),     # y axis
                     np.nanpercentile(plateau_p, 50)],2)) # pupil size
        
        self.Gaze_9_OD = np.array(Gaze_9_OD)
        self.Gaze_9_OS = np.array(Gaze_9_OS)
        
        for arg in args:
            if arg:
                Finished_ACT = True
                ACT_Task = arg
            
        for i in range(0,len(GAZE_9_TIME)):
            diff_OD_x = self.Gaze_9_OD[0][0]-self.Gaze_9_OD[i][0]
            diff_OD_y = self.Gaze_9_OD[0][1]-self.Gaze_9_OD[i][1]
            diff_OS_x = self.Gaze_9_OS[0][0]-self.Gaze_9_OS[i][0]
            diff_OS_y = self.Gaze_9_OS[0][1]-self.Gaze_9_OS[i][1]
            if Finished_ACT:
                ACTdiff_OD_x = self.Gaze_9_OD[0][0]-ACT_Task.OD_ACT[1][0]
                ACTdiff_OD_y = self.Gaze_9_OD[0][1]-ACT_Task.OD_ACT[1][1]
                ACTdiff_OS_x = self.Gaze_9_OS[0][0]-ACT_Task.OS_ACT[2][0]
                ACTdiff_OS_y = self.Gaze_9_OS[0][1]-ACT_Task.OS_ACT[2][1]
                AG_OD = trans_AG(self.AL_OD,np.array([diff_OD_x, diff_OD_y]),CAL_VAL_OD) + trans_AG(self.AL_OD,np.array([ACTdiff_OD_x, ACTdiff_OD_y]),CAL_VAL_OD)
                AG_OS = trans_AG(self.AL_OS,np.array([diff_OS_x, diff_OS_y]),CAL_VAL_OS) + trans_AG(self.AL_OS,np.array([ACTdiff_OS_x, ACTdiff_OS_y]),CAL_VAL_OS)
            else:
                AG_OD = trans_AG(self.AL_OD,np.array([diff_OD_x, diff_OD_y]),CAL_VAL_OD)
                AG_OS = trans_AG(self.AL_OS,np.array([diff_OS_x, diff_OS_y]),CAL_VAL_OS)
            NeurobitDxDev_H.append([AG_OD[0], AG_OS[0]])
            NeurobitDxDev_V.append([AG_OD[1], AG_OS[1]])
        self.NeurobitDxDev_H = np.round(np.array(NeurobitDxDev_H),2)
        self.NeurobitDxDev_V = np.round(np.array(NeurobitDxDev_V),2)
        
        x_lower = self.Gaze_9_OD.transpose()[0][[5,6,0,3,2]]
        x_upper = self.Gaze_9_OD.transpose()[0][[5,7,8,4,2]]
        y_lower = -self.Gaze_9_OD.transpose()[1][[5,6,0,3,2]]
        y_upper = -self.Gaze_9_OD.transpose()[1][[5,7,8,4,2]]
        OD_Area = enclosed_area(x_lower, y_lower, x_upper, y_upper)
        x_lower = self.Gaze_9_OS.transpose()[0][[5,6,0,3,2]]
        x_upper = self.Gaze_9_OS.transpose()[0][[5,7,8,4,2]]
        y_lower = -self.Gaze_9_OS.transpose()[1][[5,6,0,3,2]]
        y_upper = -self.Gaze_9_OS.transpose()[1][[5,7,8,4,2]]
        OS_Area = enclosed_area(x_lower, y_lower, x_upper, y_upper)
                
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(GAZE_9_TIME)):
            temp = self.CmdTime[GAZE_9_TIME[i]]
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)
            diff_temp = np.diff(temp)
            inds = np.where(diff_temp>20)[0]
            if len(inds)>0:
                list_temp = list(); j = 0
                for ind in inds:
                    list_temp.append(temp[j:ind])
                    j = ind
                list_temp.append(temp[ind:])
                self.CmdTime[GAZE_9_TIME[i]] = list_temp
            else:
                self.CmdTime[GAZE_9_TIME[i]] = temp
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]      
        for i in range(0,len(GAZE_9_TIME)):
            if frame_cnt in self.CmdTime[GAZE_9_TIME[i]]:
                text = GAZE_9_STR[i]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)[0]
                textX = int((width - textsize[0]) / 2)
                cv2.putText(frame,text, (textX, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 
                            2, (255, 255, 255),
                            2, cv2.LINE_AA)        
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/25
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(EYE)):
            if EYE[i] == 'OD':
                x_diff = self.Gaze_9_OD[0][0]-OD[0,:]
                y_diff = self.Gaze_9_OD[0][1]-OD[1,:]
                x_PD = trans_AG(self.AL_OD,x_diff,CAL_VAL_OD)
                y_PD = trans_AG(self.AL_OD,y_diff,CAL_VAL_OD)
            else:
                x_diff = self.Gaze_9_OS[0][0]-OS[0,:]
                y_diff = self.Gaze_9_OS[0][1]-OS[1,:]
                x_PD = trans_AG(self.AL_OS,x_diff,CAL_VAL_OS)
                y_PD = trans_AG(self.AL_OS,y_diff,CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position ("+chr(176)+")")
            plt.title("9 Gaze Test "+ EYE[i])
            plt.grid(True, linestyle=':')
            plt.xticks(fontsize= 8)
            plt.yticks(fontsize= 8)
            plt.ylim([-100,100])
            plt.text(0,90, "right",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(0,-90, "left",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1],90,"up",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1], -90,"down",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8)        
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300) 
        plt.close()
    def DrawEyeFig(self):
        Gaze_9 = []; OD = self.OD; OS = self.OS
        for i in range(0,len(self.Gaze_9_OD)):
            try:t = np.concatenate(np.array(self.CmdTime[GAZE_9_TIME[i]]))
            except:t = self.CmdTime[GAZE_9_TIME[i]]
            if not np.isnan(self.Gaze_9_OD[i,0]) and not np.isnan(self.Gaze_9_OS[i,0]):
                OD_diff = abs(OD[0,t]-self.Gaze_9_OD[i,0])+abs(OD[1,t]-self.Gaze_9_OD[i,1])
                OS_diff = abs(OS[0,t]-self.Gaze_9_OS[i,0])+abs(OS[1,t]-self.Gaze_9_OS[i,1])
                Diff = np.sum(np.array([OD_diff, OS_diff]),axis = 0)
            elif np.isnan(self.Gaze_9_OD[i,0]):
                Diff = abs(OS[0,t]-self.Gaze_9_OS[i,0])+abs(OS[1,t]-self.Gaze_9_OS[i,1])
            else:
                Diff = abs(OD[0,t]-self.Gaze_9_OD[i,0])+abs(OD[1,t]-self.Gaze_9_OD[i,1])                
            if not np.isnan(Diff).all():
                Gaze_9.append(t[np.where(Diff == np.nanmin(Diff))[0][0]])
            else:
                Gaze_9.append(np.nan)
        pic_cont = 0
        empt=0
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)
        for pic in Gaze_9:
            if not np.isnan(pic):
                cap = GetVideo(self.csv_path)
                cap.set(1,pic)
                ret, im = cap.read()
                height = im.shape[0]
                width = im.shape[1]
                try:
                    cv2.rectangle(im,
                                  (int(self.Gaze_9_OD[pic_cont][0]),int(self.Gaze_9_OD[pic_cont][1])),
                                  (int(self.Gaze_9_OD[pic_cont][0])+1,int(self.Gaze_9_OD[pic_cont][1])+1),
                                  (0,255,0),2)
                    cv2.circle(im,(int(self.Gaze_9_OD[pic_cont][0]),int(self.Gaze_9_OD[pic_cont][1])),
                               int(self.Gaze_9_OD[pic_cont][2]),
                               (255,255,255),2) 
                except:
                    pass#print("OD Absent!")
                try:
                    cv2.rectangle(im,
                                  (int(self.Gaze_9_OS[pic_cont][0]),int(self.Gaze_9_OS[pic_cont][1])),
                                  (int(self.Gaze_9_OS[pic_cont][0])+1,int(self.Gaze_9_OS[pic_cont][1])+1),
                                  (0,255,0),2)
                    cv2.circle(im,(int(self.Gaze_9_OS[pic_cont][0]),int(self.Gaze_9_OS[pic_cont][1])),
                               int(self.Gaze_9_OS[pic_cont][2]),
                               (255,255,255),2)
                except:
                    pass#print("OS Absent!")
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                _,thresh_1 = cv2.threshold(gray,110,255,cv2.THRESH_TRUNC)
                
                exec('ax'+str(pic_cont+1)+'=plt.subplot(3, 3, GAZE_9_EYEFIG[pic_cont])')
                exec('ax'+str(pic_cont+1)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
                exec('ax'+str(pic_cont+1)+'.axes.xaxis.set_ticks([])')
                exec('ax'+str(pic_cont+1)+ '.axes.yaxis.set_ticks([])')
                exec('ax'+str(pic_cont+1)+ '.set_ylim(int(3*height/4),int(height/4))')
# =============================================================================
#                 exec('ax'+str(pic_cont+1)+ '.set_ylim(int(height),int(0))')
# =============================================================================
                exec('ax'+str(pic_cont+1)+ '.set_ylabel(GAZE_9_TIME[pic_cont])')
                plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
        plt.close()
    def DrawEyeMesh(self):
        border = [7,2,8,4,6,1,5,3,7]
        MIN = -50; MAX = 50
        fig = plt.gcf()
        fig.set_size_inches(6/1.2,3/1.2, forward=True)
        fig.set_dpi(300)
        cir_size = 25
        try:
            diff_V = np.round(math.degrees(math.atan(abs(170-self.Height)/220)),2)
            if self.Height>170:
                diff_V = -diff_V
        except:
            diff_V = 0
        for i in range(0,len(EYE)):
            ax = plt.subplot(1,2,i+1)
            ax.xaxis.set_ticks(np.array(range(MIN,MAX,5)))
            ax.yaxis.set_ticks(np.array(range(MIN,MAX,5)))
            ax.grid(which='major') 
            ax.grid(which='minor') 
            majorLocator = MultipleLocator(25)
            minorLocator = MultipleLocator(5)
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_minor_locator(minorLocator)

            plt.vlines(0,MIN,MAX,linewidth = .5,colors = 'k')
            plt.hlines(0,MIN,MAX,linewidth = .5,colors = 'k')
            plt.vlines(20,-15,15,linewidth = .5,colors = 'g')
            plt.hlines(15,-20,20,linewidth = .5,colors = 'g')
            plt.vlines(-20,-15,15,linewidth = .5,colors = 'g')
            plt.hlines(-15,-20,20,linewidth = .5,colors = 'g')
            plt.title(EYE[i]+" (°)")
            plt.grid(True,alpha = 0.5)
            plt.scatter(-self.NeurobitDxDev_H[:,i],self.NeurobitDxDev_V[:,i]+diff_V,
                        s = cir_size,c = 'k',)
            plt.plot(-self.NeurobitDxDev_H[border,i],self.NeurobitDxDev_V[border,i]+diff_V,
                        linewidth = .5,c = 'r',)
            plt.xlim([MIN,MAX])
            plt.ylim([MIN,MAX])
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeMesh.png"), dpi=300)
        plt.close()
        
class CUT_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "CUT"
        self.sequence = 0
        self.FolderName = csv_path.split('\\')[-2]
        self.FileName = csv_path.split('\\')[-1].replace(".csv","")
        self.main_path = csv_path.replace("\\"+csv_path.split('\\')[-2],"").replace("\\"+csv_path.split('\\')[-1],"")
        self.save_MainPath = self.main_path+"\\"+self.FolderName
        self.saveReport_path = self.save_MainPath
        self.saveMerge_path = self.save_MainPath+"\\"+self.task
        self.saveVideo_path = self.save_MainPath+"\\"+self.task+"\\HoughCircle"
        self.saveImage_path = self.save_MainPath+"\\"+self.task+"\\Image"              
        if not os.path.isdir(self.saveVideo_path):
            os.makedirs(self.saveVideo_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
    def Exec(self):
        self.GetCommand()        
        self.GetEyePosition()  
        self.SeperateSession()              
        self.FeatureExtraction()  
        self.GetDiagnosis()  
        self.Save2Cloud()
        
        self.DrawEyeFig()
        self.DrawEyeTrack()  
        self.DrawQRCode()
    def FeatureExtraction(self):
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        OD_mean = np.nanmean(OD[0,:]); OD_std = np.nanstd(OD[0,:])
        OS_mean = np.nanmean(OS[0,:]); OS_std = np.nanstd(OS[0,:])
        delet_OD = np.logical_and(OD[0,:]>OD_mean+OD_std*2, OD[0,:]>OD_mean-OD_std*2)
        delet_OS = np.logical_and(OS[0,:]>OS_mean+OS_std*2, OS[0,:]>OS_mean-OS_std*2)
        if delet_OD.any():
            OD[:,delet_OD] = np.nan
        if delet_OS.any():
            OS[:,delet_OS] = np.nan
        OD_ACT = []; OS_ACT = [];       # all position in each CUT_TIME
        for i in range(0,len(CUT_TIME)):
            temp = self.CmdTime[CUT_TIME[i]]
# =============================================================================
#             delete = np.where(temp>len(OD[0])-1)[0]
#             if delete.any():
#                 temp = np.delete(temp, delete)
# =============================================================================
            if type(self.CmdTime[CUT_TIME[i]]) == list:
                OD_ACT_tmp = []; OS_ACT_tmp = [];
                thr_odx = 50; thr_ody = 50; thr_osx = 50; thr_osy = 50
                for temp in self.CmdTime[CUT_TIME[i]]:
                    diff_x = np.diff(OD[0][temp])
                    grad_x = np.sign(diff_x)
                    plateau_x = OD[0][temp][np.where(grad_x==0)[0]+1]
                    
                    diff_y = np.diff(OD[1][temp])
                    grad_y = np.sign(diff_y)
                    plateau_y = OD[1][temp][np.where(grad_y==0)[0]+1]
                    
                    diff_p = np.diff(OD[2][temp])
                    grad_p = np.sign(diff_p)
                    plateau_p = OD[2][temp][np.where(grad_p==0)[0]+1]
                    
                    OD_ACT_tmp.append(np.round(
                        [np.nanpercentile(plateau_x, 50),     # x axis
                         np.nanpercentile(plateau_y, 50),     # y axis
                         np.nanpercentile(plateau_p, 50)],2)) # pupil size
                    
                    diff_x = np.diff(OS[0][temp])
                    grad_x = np.sign(diff_x)
                    plateau_x = OS[0][temp][np.where(grad_x==0)[0]+1]
                    
                    diff_y = np.diff(OS[1][temp])
                    grad_y = np.sign(diff_y)
                    plateau_y = OS[1][temp][np.where(grad_y==0)[0]+1]
                    
                    diff_p = np.diff(OS[2][temp])
                    grad_p = np.sign(diff_p)
                    plateau_p = OS[2][temp][np.where(grad_p==0)[0]+1]
                    
                    OS_ACT_tmp.append(np.round(
                        [np.nanpercentile(plateau_x, 50),     # x axis
                         np.nanpercentile(plateau_y, 50),     # y axis
                         np.nanpercentile(plateau_p, 50)],2)) # pupil size
                    
                OD_ACT_tmp = np.array(OD_ACT_tmp)
                OS_ACT_tmp = np.array(OS_ACT_tmp)
                OD_ACT.append(np.round(
                    [np.nanpercentile(OD_ACT_tmp[:,0], 50),     # x axis
                     np.nanpercentile(OD_ACT_tmp[:,1], 50),     # y axis
                     np.nanpercentile(OD_ACT_tmp[:,2], 50)],2))
                OS_ACT.append(np.round(
                    [np.nanpercentile(OS_ACT_tmp[:,0], 50),     # x axis
                     np.nanpercentile(OS_ACT_tmp[:,1], 50),     # y axis
                     np.nanpercentile(OS_ACT_tmp[:,2], 50)],2))
            else:
                diff_x = np.diff(OD[0][temp])
                grad_x = np.sign(diff_x)
                plateau_x = OD[0][temp][np.where(grad_x==0)[0]+1]
                
                diff_y = np.diff(OD[1][temp])
                grad_y = np.sign(diff_y)
                plateau_y = OD[1][temp][np.where(grad_y==0)[0]+1]
                
                diff_p = np.diff(OD[2][temp])
                grad_p = np.sign(diff_p)
                plateau_p = OD[2][temp][np.where(grad_p==0)[0]+1]
                
                OD_ACT.append(np.round(
                    [np.nanpercentile(plateau_x, 50),     # x axis
                     np.nanpercentile(plateau_y, 50),     # y axis
                     np.nanpercentile(plateau_p, 50)],2)) # pupil size
                diff_x = np.diff(OS[0][temp])
                grad_x = np.sign(diff_x)
                plateau_x = OS[0][temp][np.where(grad_x==0)[0]+1]
                
                diff_y = np.diff(OS[1][temp])
                grad_y = np.sign(diff_y)
                plateau_y = OS[1][temp][np.where(grad_y==0)[0]+1]
                
                diff_p = np.diff(OS[2][temp])
                grad_p = np.sign(diff_p)
                plateau_p = OS[2][temp][np.where(grad_p==0)[0]+1]
                
                OS_ACT.append(np.round(
                    [np.nanpercentile(plateau_x, 50),     # x axis
                     np.nanpercentile(plateau_y, 50),     # y axis
                     np.nanpercentile(plateau_p, 50)],2)) # pupil size
        
        # ET、XT angle
        OD_ACT = np.array(OD_ACT)
        OS_ACT = np.array(OS_ACT)
        self.OD_ACT = OD_ACT
        self.OS_ACT = OS_ACT
        # Fixation_eye - Uncovered_eye
        OS_phoria = OS_ACT[2]-OS_ACT[1]    # UCL-CL
        OD_phoria = OD_ACT[4]-OD_ACT[3]    # UCR-CR
        
        OD_fix = OD_ACT[1]-OD_ACT[3]    # CL-CR
        OS_fix = OS_ACT[3]-OS_ACT[1]    # CR-CL
        
        try:
            OD_fix = np.append(trans_PD(self.AL_OD,OD_fix[0:2],CAL_VAL_OD), OD_fix[2])
            OS_fix = np.append(trans_PD(self.AL_OS,OS_fix[0:2],CAL_VAL_OS), OS_fix[2])
            
            OS_phoria = np.append(trans_PD(self.AL_OS,OS_phoria[0:2],CAL_VAL_OS), OS_phoria[2])
            OD_phoria = np.append(trans_PD(self.AL_OD,OD_phoria[0:2],CAL_VAL_OD), OD_phoria[2])
        except:
            pass#print("No profile")
        self.OD_fix = OD_fix        # one position in each CUT_TIME
        self.OS_fix = OS_fix
        self.OD_phoria = OD_phoria        # one position in each CUT_TIME
        self.OS_phoria = OS_phoria
    def GetDiagnosis(self):
        OD_fix = self.OD_fix; OS_fix = self.OS_fix
        OD_phoria = self.OD_phoria; OS_phoria = self.OS_phoria
        thr =1.5
        self.NeurobitDx_H = None
        self.NeurobitDx_V = None
        self.NeurobitDxTp_X = None
        if np.all(np.abs([OD_fix,OS_fix,OS_phoria,OD_phoria])<=thr):
            self.Ortho = True
            self.NeurobitDx_H = 'Ortho'
            self.NeurobitDx_V = 'Ortho'
            self.NeurobitDxTp_H = 'None'
            self.NeurobitDxDev_H = 0
            self.NeurobitDxDev_V = 0
        else:
            self.Ortho = False
        """Tropia"""
        if -OS_fix[0]>thr or OD_fix[0]>thr:
            self.NeurobitDx_H = 'XT'
            if -OS_fix[0]>thr and OD_fix[0]>thr:
                self.NeurobitDxTp_H = 'Divergence'
                self.NeurobitDxDev_H = (abs(OD_fix[0])+abs(OS_fix[0]))/2
            elif -OS_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OS_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OS, Levoversion'
                else:
                    self.NeurobitDxTp_X = 'OS, Divergence'
            elif OD_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OD_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OD, Levoversion'
                else:
                    self.NeurobitDxTp_X = 'OD, Divergence'            
        elif OS_fix[0]>thr or -OD_fix[0]>thr:
            self.NeurobitDx_H = 'ET'
            if OS_fix[0]>thr and -OD_fix[0]>thr:
                self.NeurobitDxTp_H = 'Convergence'
                self.NeurobitDxDev_H = (abs(OD_fix[0])+abs(OS_fix[0]))/2
            elif OS_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OS_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OS Detroversion'
                else:
                    self.NeurobitDxTp_X = 'OS Convergence'           
            elif -OD_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OD_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OD Detroversion'
                else:
                    self.NeurobitDxTp_X = 'OD Convergence'
        else:
            self.NeurobitDx_H = 'Ortho'
            self.NeurobitDxDev_H = 0
        
        """Phoria"""
        if OS_phoria[0]>thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OS XP"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OS_phoria[0]))
        elif OS_phoria[0]<-thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OS E"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OS_phoria[0]))
        else: self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H,0)
            
        if OD_phoria[0]>thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OD XP"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OD_phoria[0]))
        elif OD_phoria[0]<-thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OD E"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OD_phoria[0]))
        else: self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H,0)
        
        """Vertical"""
        if OS_fix[1]>thr or -OD_fix[1]>thr:
            self.NeurobitDx_V = 'LHT'
            if OS_fix[1]>thr:
                self.NeurobitDxDev_V = abs(OS_fix[1])
            else:
                self.NeurobitDxDev_V = abs(OD_fix[1])
        
        elif -OS_fix[1]>thr or OD_fix[1]>thr:
            self.NeurobitDx_V = 'LHoT'
            if OS_fix[1]>thr:
                self.NeurobitDxDev_V = abs(OS_fix[1])
            else:
                self.NeurobitDxDev_V = abs(OD_fix[1])
        else:
            self.NeurobitDx_V = 'Ortho'
            self.NeurobitDxDev_V = 0    
        
        if OD_phoria[1]>thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OD HP"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OD_phoria[1]))
        elif OD_phoria[1]<-thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OD H"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OD_phoria[1]))
        else: self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V,0)
        
        if OS_phoria[1]>thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OS HP"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OS_phoria[1]))
        elif OS_phoria[1]<-thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OS H"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OS_phoria[1]))
        else: self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V,0)
   
    def GetTimeFromCmd(self):
        cmd = self.VoiceCommand
        O_t = np.where(cmd==0)[0]
        CL_t = np.where(cmd==1)[0]
        UCL_t = np.where(cmd==2)[0]
        CR_t = np.where(cmd==3)[0] 
        UCR_t = np.where(cmd==4)[0]
        self.CmdTime = {"CL_t": np.array(CL_t),
                        "UCL_t": np.array(UCL_t),
                        "CR_t": np.array(CR_t),
                        "UCR_t": np.array(UCR_t),
                        "O_t":  np.array(O_t)}
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(CUT_TIME)):
            temp = np.array(self.CmdTime[CUT_TIME[i]])
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)
            diff_temp = np.diff(temp)
            inds = np.where(diff_temp>20)[0]
            if len(inds)>0:
                list_temp = list(); j = 0
                for ind in inds:
                    list_temp.append(temp[j:ind])
                    j = ind
                list_temp.append(temp[ind:])
                self.CmdTime[CUT_TIME[i]] = list_temp
            else:
                self.CmdTime[CUT_TIME[i]] = temp
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/25
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(EYE)):
            if EYE[i] == 'OD':
                x_diff = self.OD_ACT[0,0]-OD[0,:]
                y_diff = self.OD_ACT[0,1]-OD[1,:]
                x_PD = trans_PD(self.AL_OD,x_diff,CAL_VAL_OD)
                y_PD = trans_PD(self.AL_OD,y_diff,CAL_VAL_OD)
            else:
                x_diff = self.OS_ACT[0,0]-OS[0,:]
                y_diff = self.OS_ACT[0,1]-OS[1,:]
                x_PD = trans_PD(self.AL_OS,x_diff,CAL_VAL_OS)
                y_PD = trans_PD(self.AL_OS,y_diff,CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position (PD)")
            plt.title("Cover Uncover Test "+ EYE[i])
            
            plt.grid(True, linestyle=':')
            plt.xticks(fontsize= 8)
            plt.yticks(fontsize= 8)
            
            plt.text(0,90, "right",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(0,-90, "left",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1], 90,"up",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1], -90,"down",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8) 
            plt.ylim([-100,100])
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300) 
    def DrawEyeFig(self):
        ACT = []; OD = self.OD; OS = self.OS
        for i in range(0,len(self.OS_ACT)):
            try:t = np.concatenate(np.array(self.CmdTime[CUT_TIME[i]]))
            except:t = self.CmdTime[CUT_TIME[i]]
            if not np.isnan(self.OD_ACT[i,0]) and not np.isnan(self.OS_ACT[i,0]):
                OD_diff = abs(OD[0,t]-self.OD_ACT[i,0])+abs(OD[1,t]-self.OD_ACT[i,1])
                OS_diff = abs(OS[0,t]-self.OS_ACT[i,0])+abs(OS[1,t]-self.OS_ACT[i,1])
                Diff = np.sum(np.array([OD_diff, OS_diff]),axis = 0)
                pupil = OS[2,t]+OD[2,t]
            elif np.isnan(self.OD_ACT[i,0]):
                Diff = abs(OS[0,t]-self.OS_ACT[i,0])+abs(OS[1,t]-self.OS_ACT[i,1])
                pupil = OS[2,t]
            else:
                Diff = abs(OD[0,t]-self.OD_ACT[i,0])+abs(OD[1,t]-self.OD_ACT[i,1])
                pupil = OD[2,t]
            try:
# =============================================================================
#                 ind = np.where(Diff == np.nanmin(Diff))[0]
#                 ind_pu = np.where(pupil[ind] == np.nanmax(pupil[ind]))[0]
# =============================================================================
                ACT.append(t[np.where(Diff == np.nanmin(Diff))[0][0]])
            except:
                ACT.append(ACT[-1])
                #print("Not Detect "+ CUT_TIME[i]) 
        pic_cont = 1
        empt=0
        #fig = plt.figure(figsize=(11.7,8.3))
        fig = plt.gcf()
        fig.set_size_inches(3,5, forward=True)
        fig.set_dpi(300)
        for pic in ACT:
            cap = GetVideo(self.csv_path)
            cap.set(1,pic)
            ret, im = cap.read()
            height = im.shape[0]
            width = im.shape[1]
            try:
                cv2.rectangle(im,
                              (int(self.OD_ACT[pic_cont-1][0]),int(self.OD_ACT[pic_cont-1][1])),
                              (int(self.OD_ACT[pic_cont-1][0])+1,int(self.OD_ACT[pic_cont-1][1])+1),
                              (0,255,0),2)
                cv2.circle(im,(int(self.OD_ACT[pic_cont-1][0]),int(self.OD_ACT[pic_cont-1][1])),
                           int(self.OD_ACT[pic_cont-1][2]),
                           (255,255,255),2) 
            except:
                pass#print("OD Absent!")
            try:
                cv2.rectangle(im,
                              (int(self.OS_ACT[pic_cont-1][0]),int(self.OS_ACT[pic_cont-1][1])),
                              (int(self.OS_ACT[pic_cont-1][0])+1,int(self.OS_ACT[pic_cont-1][1])+1),
                              (0,255,0),2)
                cv2.circle(im,(int(self.OS_ACT[pic_cont-1][0]),int(self.OS_ACT[pic_cont-1][1])),
                           int(self.OS_ACT[pic_cont-1][2]),
                           (255,255,255),2)
            except:
                pass#print("OS Absent!")
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _,thresh_1 = cv2.threshold(gray,110,255,cv2.THRESH_TRUNC)
            exec('ax'+str(pic_cont)+'=plt.subplot(5, 1, pic_cont)')
            exec('ax'+str(pic_cont)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
            exec('ax'+str(pic_cont)+'.axes.xaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.axes.yaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.set_ylim(int(3*height/4),int(height/4))')
# =============================================================================
#             exec('ax'+str(pic_cont)+ '.set_ylim(int(height),int(0))')
# =============================================================================
            exec('ax'+str(pic_cont)+ '.set_ylabel(CUT_LABEL[pic_cont-1])')
            plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]
        for i in range(0,len(CUT_TIME)):
            if frame_cnt in self.CmdTime[CUT_TIME[i]]:
                text = CUT_STR[i]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)[0]
                textX = int((width - textsize[0]) / 2)
                cv2.putText(frame,text, (textX, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 
                            2, (255, 255, 255),
                            2, cv2.LINE_AA)
        if self.IsVoiceCommand:
            text = "Voice Command"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)[0]
            textX = int((width - textsize[0]) / 2)
            cv2.putText(frame,text, (textX, 550), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (0, 255, 255),
                        1, cv2.LINE_AA)
        else:
            text = "No Voice Command"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)[0]
            textX = int((width - textsize[0]) / 2)
            cv2.putText(frame,text, (textX, 550), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (0, 255, 255),
                        1, cv2.LINE_AA)         
        
        