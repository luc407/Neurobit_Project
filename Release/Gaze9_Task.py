# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:38:32 2022

@author: luc40
"""
import os
import cv2
import math
import tkinter as tk
import numpy as np
import Neurobit as nb
from Neurobit import Neurobit
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
# =============================================================================
# from EyeTrack import EyeTrackSys
# =============================================================================
    
class Gaze9_Task(Neurobit):
    def __init__(self, csv_path):
        nb.Neurobit.__init__(self)
        self.task = "9_Gaze"
        self.Mode = "OcularMotility"
        self.FolderName = csv_path.split('\\')[-2]
        self.FileName = csv_path.split('\\')[-1].replace(".csv","")
        self.main_path = csv_path.replace("\\"+csv_path.split('\\')[-2],"").replace("\\"+csv_path.split('\\')[-1],"")
        self.save_MainPath = self.save_path+"\\"+self.FolderName
        self.saveReport_path = self.main_path+"\\"+self.FolderName
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
# =============================================================================
#         self.Preprocessing()
# =============================================================================
        #self.SeperateSession()                
        if Finished_ACT: self.FeatureExtraction(ACT_Task) 
        else: self.FeatureExtraction() 
# =============================================================================
#         self.Save2Cloud()
# =============================================================================
               
        self.DrawEyeFig()
        self.DrawEyeMesh()
        self.DrawEyeTrack()  
# =============================================================================
#         self.DrawQRCode() 
# =============================================================================
    def GetTimeFromCmd(self):        
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
        
        for i in range(0,len(nb.GAZE_9_TIME)):
            exec('cov_non = np.where(cover['+nb.GAZE_9_TIME[i]+'] >= 0)[0]')
            exec('self.CmdTime[nb.GAZE_9_TIME[i]] = '+nb.GAZE_9_TIME[i]+'[cov_non]')
    def FeatureExtraction(self,*args):
        def GetTrialCmd(i,eyePosition,Gaze9,T,CmdTime,AdjCmdTime):
            window = int(3*30) # respond period = seconds*fps 
            LT = 10  # Set default latency
            CmdTmp = CmdTime[T];print(T,i)            
            
            Trial_trg_ind = np.where(np.diff(CmdTime[T]) > 5)[0]
            start_ind = np.where(CmdTmp == i)[0][0]
            end_ind = Trial_trg_ind[np.where(Trial_trg_ind>start_ind)[0]]
            if end_ind.any(): end_ind = end_ind[0]
            else: end_ind = len(CmdTmp)-1            
            
            t = CmdTmp[start_ind]-1;            
            baseline_x = np.nanmean(eyePosition[0,CmdTmp[start_ind]:CmdTmp[start_ind]+LT]) # mean value in latency
            baseline_y = np.nanmean(eyePosition[1,CmdTmp[start_ind]:CmdTmp[start_ind]+LT]) # mean value in latency
            
            tmp_t = []; 
            if T == 'F':                
                try:
                    Gaze9[T] = np.append(Gaze9[T], [np.nanpercentile(eyePosition[:,CmdTmp[start_ind:end_ind]],50,axis = 1)],axis = 0)
                except:
                    Gaze9[T] = [np.nanpercentile(eyePosition[:,CmdTmp[start_ind:end_ind]],50,axis = 1)]
                try:
                    AdjCmdTime[T] = np.append(AdjCmdTime[T],CmdTmp[end_ind])
                except:
                    AdjCmdTime[T] = CmdTmp[end_ind]
            elif T == 'U':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if y < pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'D':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if y > pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'R':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x < pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'L':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x > pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'RU':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x < pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'LU':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x > pre_x or y < pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'RD':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x < pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'LD':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x > pre_x or y > pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            if not tmp_t: 
                tmp_t = CmdTmp[start_ind]
                fin_x, fin_y, fin_p = list(np.nanmean(eyePosition[:,CmdTmp[start_ind:end_ind]],axis = 1))
            if T != 'F':
                try:
                    Gaze9[T] = np.append(Gaze9[T], [[fin_x,fin_y,fin_p]],axis = 0)
                except:
                    Gaze9[T] = [[fin_x,fin_y,fin_p]]
                try:
                    AdjCmdTime[T] = np.append(AdjCmdTime[T],tmp_t)
                except:
                    AdjCmdTime[T] = tmp_t
            i = CmdTmp[end_ind]+1
            return Gaze9, AdjCmdTime, i
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
                
        AdjCmdTime = dict()
        OD_Gaze9 = dict()
        OS_Gaze9 = dict()
                
        i = 0; PASS = False;      # read CmdTime        
        while(i < len(OD[0,:])):
            for TIME in range(0,len(nb.GAZE_9_TIME)):                
                if i in self.CmdTime[nb.GAZE_9_TIME[TIME]]:
                    T = nb.GAZE_9_TIME[TIME]
                    OD_Gaze9,AdjCmdTime,_ = GetTrialCmd(i,OD,OD_Gaze9,T,self.CmdTime,AdjCmdTime)
                    OS_Gaze9,AdjCmdTime,i = GetTrialCmd(i,OS,OS_Gaze9,T,self.CmdTime,AdjCmdTime)
                    PASS = True
                    break
                else:
                    PASS = False                
            if not PASS:
                i+=1
        
        #GAZE_9_TIME     = ['F','U','D','R','L','RU','LU','RD','LD']
        OD = self.OD; OS = self.OS; Finished_ACT = False
        Gaze_9_OD = []; Gaze_9_OS = [];       # all position in each ACT_TIME
        NeurobitDxDev_H = list();NeurobitDxDev_V = list()
        for i in range(0,len(nb.GAZE_9_TIME)):
            if AdjCmdTime[nb.GAZE_9_TIME[i]].any():
                T = nb.GAZE_9_TIME[i]
                if T == 'F':
                    Gaze_9_OD.append(np.nanpercentile(OD_Gaze9[T],50,axis = 0))
                    Gaze_9_OS.append(np.nanpercentile(OS_Gaze9[T],50,axis = 0))
                else:
                    if len(OD_Gaze9[T])>1:
                        diff = abs(OD_Gaze9[T][:,0]-Gaze_9_OD[0][0])+abs(OD_Gaze9[T][:,1]-Gaze_9_OD[0][1])
                        loc = np.where(diff == np.max(diff))[0][0]
                        Gaze_9_OD.append(OD_Gaze9[T][loc,:])
                        Gaze_9_OS.append(OS_Gaze9[T][loc,:])
                    else:
                        Gaze_9_OD.append(OD_Gaze9[T][0])
                        Gaze_9_OS.append(OS_Gaze9[T][0])
                    
                                        
        self.Gaze_9_OD = np.array(np.round(Gaze_9_OD,2))
        self.Gaze_9_OS = np.array(np.round(Gaze_9_OS,2))
        self.CmdTime = AdjCmdTime
        
        for arg in args:
            if arg:
                Finished_ACT = True
                ACT_task = arg
            
        for i in range(0,len(nb.GAZE_9_TIME)):
            if Finished_ACT:
                diff_OD_x = ACT_task.OD_ACT[1][0]-self.Gaze_9_OD[i][0]
                diff_OD_y = ACT_task.OD_ACT[1][1]-self.Gaze_9_OD[i][1]
                diff_OS_x = ACT_task.OS_ACT[2][0]-self.Gaze_9_OS[i][0]
                diff_OS_y = ACT_task.OS_ACT[2][1]-self.Gaze_9_OS[i][1]
            else:
                F_t = [i for i in range(0,len(nb.GAZE_9_TIME)) if nb.GAZE_9_TIME[i] == "F"][0]
                diff_OD_x = self.Gaze_9_OD[F_t][0]-self.Gaze_9_OD[i][0]
                diff_OD_y = self.Gaze_9_OD[F_t][1]-self.Gaze_9_OD[i][1]
                diff_OS_x = self.Gaze_9_OS[F_t][0]-self.Gaze_9_OS[i][0]
                diff_OS_y = self.Gaze_9_OS[F_t][1]-self.Gaze_9_OS[i][1]
            PD_OD = nb.trans_AG(self.AL_OD,np.array([diff_OD_x, diff_OD_y]),nb.CAL_VAL_OD)
            PD_OS = nb.trans_AG(self.AL_OS,np.array([diff_OS_x, diff_OS_y]),nb.CAL_VAL_OS)
            NeurobitDxDev_H.append([PD_OD[0], PD_OS[0]])
            NeurobitDxDev_V.append([PD_OD[1], PD_OS[1]])
        self.NeurobitDxDev_H = np.round(np.array(NeurobitDxDev_H),2)
        self.NeurobitDxDev_V = np.round(np.array(NeurobitDxDev_V),2)
        
        #GAZE_9_TIME     = ['F','U','D','R','L','RU','LU','RD','LD']
        a = self.NeurobitDxDev_H.transpose()[0][nb.GAZE_9_BOARDER]
        b = self.NeurobitDxDev_V.transpose()[0][nb.GAZE_9_BOARDER]
        xy = np.array([a,b]).transpose()
        OD_Area = np.round(nb.enclosed_area(xy),2)
        
        a = self.NeurobitDxDev_H.transpose()[1][nb.GAZE_9_BOARDER]
        b = self.NeurobitDxDev_V.transpose()[1][nb.GAZE_9_BOARDER]
        xy = np.array([a,b]).transpose()
        OS_Area = np.round(nb.enclosed_area(xy),2)
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(nb.GAZE_9_TIME)):
            temp = self.CmdTime[nb.GAZE_9_TIME[i]]
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
                self.CmdTime[nb.GAZE_9_TIME[i]] = list_temp
            else:
                self.CmdTime[nb.GAZE_9_TIME[i]] = temp
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]      
        for i in range(0,len(nb.GAZE_9_TIME)):
            if frame_cnt in self.CmdTime[nb.GAZE_9_TIME[i]]:
                text = nb.GAZE_9_STR[i]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)[0]
                textX = int((width - textsize[0]) / 2)
                cv2.putText(frame,text, (textX, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 
                            2, (255, 255, 255),
                            2, cv2.LINE_AA)        
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/30
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(nb.EYE)):
            if nb.EYE[i] == 'OD':
                x_diff = self.Gaze_9_OD[0][0]-OD[0,:]
                y_diff = self.Gaze_9_OD[0][1]-OD[1,:]
                x_PD = nb.trans_AG(self.AL_OD,x_diff,nb.CAL_VAL_OD)
                y_PD = nb.trans_AG(self.AL_OD,y_diff,nb.CAL_VAL_OD)
            else:
                x_diff = self.Gaze_9_OS[0][0]-OS[0,:]
                y_diff = self.Gaze_9_OS[0][1]-OS[1,:]
                x_PD = nb.trans_AG(self.AL_OS,x_diff,nb.CAL_VAL_OS)
                y_PD = nb.trans_AG(self.AL_OS,y_diff,nb.CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position ("+chr(176)+")")
            plt.title("9 Gaze Test "+ nb.EYE[i])
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
            try:t = np.concatenate(np.array(self.CmdTime[nb.GAZE_9_TIME[i]]))
            except:t = self.CmdTime[nb.GAZE_9_TIME[i]]

            delete = np.where(t>len(OD[0])-1)[0]
            if delete.any():
                t = np.delete(t, delete)
                
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
        fig.set_size_inches(10,3.125, forward=True)
        fig.set_dpi(300)
        for pic in Gaze_9:
            if not np.isnan(pic):
                cap = nb.GetVideo(self.csv_path)
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
                
                exec('ax'+str(pic_cont+1)+'=plt.subplot(3, 3, nb.GAZE_9_EYEFIG[pic_cont])')
                exec('ax'+str(pic_cont+1)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
                exec('ax'+str(pic_cont+1)+'.axes.xaxis.set_ticks([])')
                exec('ax'+str(pic_cont+1)+ '.axes.yaxis.set_ticks([])')
                exec('ax'+str(pic_cont+1)+ '.set_ylim(int(9*height/10),int(height/8))')
# =============================================================================
#                 exec('ax'+str(pic_cont+1)+ '.set_ylim(int(height),int(0))')
# =============================================================================
                exec('ax'+str(pic_cont+1)+ '.set_ylabel(nb.GAZE_9_TIME[pic_cont])')
                plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
        plt.close()
    def DrawEyeMesh(self):
        border = np.concatenate((nb.GAZE_9_BOARDER, [nb.GAZE_9_BOARDER[0]]))
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
        for i in range(0,len(nb.EYE)):
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
            plt.title(nb.EYE[i]+" (°)")
            plt.grid(True,alpha = 0.5)
            plt.scatter(self.NeurobitDxDev_H[:,i],self.NeurobitDxDev_V[:,i]+diff_V,
                        s = cir_size,c = 'k',)
            plt.plot(self.NeurobitDxDev_H[border,i],self.NeurobitDxDev_V[border,i]+diff_V,
                        linewidth = .5,c = 'r',)
            plt.xlim([MIN,MAX])
            plt.ylim([MIN,MAX])
            ax.invert_xaxis()
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeMesh.png"), dpi=300)
        plt.close()
        