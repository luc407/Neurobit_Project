# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:37:16 2022

@author: luc40
"""
import os
import cv2
import numpy as np
import Neurobit as nb
from Neurobit import Neurobit
from scipy import stats
from matplotlib import pyplot as plt
    
class ACT_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "ACT"
        self.Mode = "OcularMotility"
        self.sequence = 0
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
    def Exec(self):
        self.GetCommand()   
        
        self.GetEyePosition()
        if not self.NoEyes:
            self.Preprocessing()
        #self.SeperateSession()              
            self.FeatureExtraction()  
            self.GetDiagnosis()  
# =============================================================================
#         self.Save2Cloud()
# =============================================================================
        
            self.DrawEyeFig()
            self.DrawEyeTrack()  
# =============================================================================
#         self.DrawQRCode()
# =============================================================================
    def NoVoiceCommandFeatureExtraction(self):
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        OD_ACT = []; OS_ACT = [];       # all position in each ACT_TIME  
        tmp = np.concatenate([np.array(self.CmdTime[nb.ACT_TIME[1]]),
                        np.array(self.CmdTime[nb.ACT_TIME[2]])])

        up_OD = np.nanpercentile(OD[:,tmp].astype(int),80, axis =1)
        low_OD = np.nanpercentile(OD[:,tmp].astype(int),20, axis =1)
        
        up_OS = np.nanpercentile(OS[:,tmp].astype(int),80, axis =1)
        low_OS = np.nanpercentile(OS[:,tmp].astype(int),20, axis =1)
        
        for i in range(0,len(nb.ACT_TIME)):
            temp = self.CmdTime[nb.ACT_TIME[i]]
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)                
            if temp.any():                
                OD_ACT.append(stats.mode(OD[:,temp].astype(int),axis = 1)[0].reshape(-1))
                OS_ACT.append(stats.mode(OS[:,temp].astype(int),axis = 1)[0].reshape(-1))
            else:
                OD_ACT.append([np.nan, np.nan, np.nan])
                OS_ACT.append([np.nan, np.nan, np.nan])
        
        CL = OD_ACT[1][:2]; CR = OD_ACT[2][:2]
        if (np.nansum(abs(CL-CR))>6):
            if sum(abs(CL-up_OD[:2])) < sum(abs(CR-up_OD[:2])):
                OD_ACT[1] = up_OD; OD_ACT[2] = low_OD
            else:
                OD_ACT[2] = up_OD; OD_ACT[1] = low_OD
            
        CL = OS_ACT[1][:2]; CR = OS_ACT[2][:2]    
        if (np.nansum(abs(CL-CR))>6):
            if sum(abs(CL-up_OS[:2])) < sum(abs(CR-up_OS[:2])):
                OS_ACT[1] = up_OS; OS_ACT[2] = low_OS
            else:
                OS_ACT[2] = up_OS; OS_ACT[1] = low_OS
        
        # ET、XT angle
        self.OD_ACT = np.array(np.round(OD_ACT,2))
        self.OS_ACT = np.array(np.round(OS_ACT,2))
        
        # Fixation_eye - Covered_eye
        OD_fix = self.OD_ACT[1]-self.OD_ACT[2]    # CL-CR
        OS_fix = self.OS_ACT[2]-self.OS_ACT[1]    # CR-CL
        
        try:
            OD_fix = np.append(nb.trans_PD(self.AL_OD,OD_fix[0:2],nb.CAL_VAL_OD), OD_fix[2])
            OS_fix = np.append(nb.trans_PD(self.AL_OS,OS_fix[0:2],nb.CAL_VAL_OS), OS_fix[2])
        except:
            print("No profile")
            
        self.OD_fix = OD_fix        # one position in each ACT_TIME
        self.OS_fix = OS_fix        
    def VoiceCommandFeatureExtraction(self):
        def GetTrialCmd(i,eyePosition,ACT_Trial,act_time,CmdTime):
            duration = int(3*30) # respond period = seconds*fps            
            LT = 5  # Set default latency
            CmdTmp = CmdTime[act_time];print(act_time, i)
            Trial_trg_ind = np.where(np.diff(CmdTime[act_time]) > 5)[0]
            start_ind = np.where(CmdTmp == i)[0][0]
            end_ind = Trial_trg_ind[np.where(Trial_trg_ind>start_ind)[0]]                
            if end_ind.any(): end_ind = end_ind[0]
            else: end_ind = len(CmdTmp)-1                
            # Find eyePosition inital respond point
            baseline = np.nanmean(eyePosition[:2,CmdTmp[start_ind:start_ind+LT]],axis = 1).reshape(2,-1) # mean value in latency
            slope = np.nansum(abs(eyePosition[:2,CmdTmp[start_ind:start_ind+duration]]-baseline),axis=0)            # normalize eye position after latency
            trg_ind = np.where(slope>2.5)[0]    # find differential more than 2.5 pixel
            
            if trg_ind.any(): 
                trg_ind = CmdTmp[start_ind] + trg_ind[0]
                ACT_Trial.append(np.nanmean(eyePosition[:,trg_ind:trg_ind+duration],axis = 1))
            else:
                ACT_Trial.append(np.nanmean(eyePosition[:,CmdTmp],axis = 1)) 
            i = CmdTmp[end_ind]+1
            return ACT_Trial, i
        
        # eyePosition
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        OD_ACT = []; OS_ACT = [];       # all position in each ACT_TIME  
        
        # ACT_Trial
        OD_ACT_CL = []; OS_ACT_CL = [];
        OD_ACT_CR = []; OS_ACT_CR = [];
        
        # Trial_trg_ind
        O_trg_ind = np.where(np.diff(self.CmdTime['O_t']) > 5)[0]
        UCR_trg_ind = np.where(np.diff(self.CmdTime['UCR_t']) > 5)[0]
        
        i = 0;      # read CmdTime
        rd = 0;     # read O_trg_ind index
        rd_ucr = 0  # read UCR_trg_ind index        
        while(i < len(OD[0,:])):
            if i in self.CmdTime['O_t']:
                tmp = self.CmdTime['O_t']
                if rd > 0 and rd < len(O_trg_ind):  # 2~(total session - 1)
                    i = tmp[O_trg_ind[rd]]+1;        rd += 1
                    continue
                elif rd > 0:    # the last session
                    i = self.CmdTime['O_t'][-1]+1;   rd += 1
                    continue
                OD_ACT.append(stats.mode(OD[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                OS_ACT.append(stats.mode(OS[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                
                i = self.CmdTime['CL_t'][0]            
                rd += 1
                print("O_t",i)
            elif i in self.CmdTime['CL_t']:
                OD_ACT_CL,_ = GetTrialCmd(i,OD,OD_ACT_CL,'CL_t',self.CmdTime)
                OS_ACT_CL,i = GetTrialCmd(i,OS,OS_ACT_CL,'CL_t',self.CmdTime)                    
            elif i in self.CmdTime['CR_t']:
                OD_ACT_CR,_ = GetTrialCmd(i,OD,OD_ACT_CR,'CR_t',self.CmdTime)
                OS_ACT_CR,i = GetTrialCmd(i,OS,OS_ACT_CR,'CR_t',self.CmdTime) 
            elif i in self.CmdTime['UCR_t']:
                print("UCR",i)
                tmp = self.CmdTime['UCR_t']
                if rd_ucr > 0 and rd_ucr < len(UCR_trg_ind): 
                    i = tmp[UCR_trg_ind[rd_ucr]]+1;        rd_ucr += 1
                    continue
                elif rd_ucr > 0:
                    i = self.CmdTime['UCR_t'][-1]+1;     rd_ucr += 1
                    continue
                OD_ACT.append(stats.mode(OD[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                OS_ACT.append(stats.mode(OS[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                
                if UCR_trg_ind.any(): i = tmp[UCR_trg_ind[rd_ucr]]+1
                else: i = self.CmdTime['UCR_t'][-1]+1                
                rd_ucr += 1                
            else:
                i+=1
        OD_ACT_CL_n = stats.mode(OD_ACT_CL,axis = 0)[0].reshape(-1)
        OS_ACT_CL_n = stats.mode(OS_ACT_CL,axis = 0)[0].reshape(-1)
        OD_ACT_CR_n = stats.mode(OD_ACT_CR,axis = 0)[0].reshape(-1)
        OS_ACT_CR_n = stats.mode(OS_ACT_CR,axis = 0)[0].reshape(-1)
        # ET、XT angle
        OD_ACT = np.insert(OD_ACT,1,OD_ACT_CL_n, axis=0)
        OD_ACT = np.insert(OD_ACT,2,OD_ACT_CR_n, axis=0)
        OS_ACT = np.insert(OS_ACT,1,OS_ACT_CL_n, axis=0)
        OS_ACT = np.insert(OS_ACT,2,OS_ACT_CR_n, axis=0)
        self.OD_ACT = np.array(np.round(OD_ACT,2))
        self.OS_ACT = np.array(np.round(OS_ACT,2))   
        
        # Fixation_eye - Covered_eye
        OD_fix_tmp = []; OS_fix_tmp = []
        for i in range(0,len(OD_ACT_CR)):
            try: OD_fix_tmp.append(OD_ACT_CL[2*i]-OD_ACT_CR[i])   # OD: CL0-CR0/ CL2-CR1 ....
            except: pass
            try: OD_fix_tmp.append(OD_ACT_CL[2*i+1]-OD_ACT_CR[i])   # OD: CL1-CR0/ CL3-CR1 ....
            except: pass
            try: OS_fix_tmp.append(OS_ACT_CR[i]-OS_ACT_CL[2*i])   # OS: CR0-CL0/ CR1-CL2 ....
            except: pass
            try: OS_fix_tmp.append(OS_ACT_CR[i]-OS_ACT_CL[2*i+1])   # OS: CR0-CL1/ CR1-CL3 ....
            except: pass
            
        OD_fix = stats.mode(np.array(np.round(OD_fix_tmp,0)).astype(int),axis = 0)[0].reshape(-1)    # CL-CR
        OS_fix = stats.mode(np.array(np.round(OS_fix_tmp,0)).astype(int),axis = 0)[0].reshape(-1)     # CR-CL
        
        # Transfer to PD unit
        try:
            OD_fix = np.append(nb.trans_PD(self.AL_OD,OD_fix[0:2],nb.CAL_VAL_OD), OD_fix[2])
            OS_fix = np.append(nb.trans_PD(self.AL_OS,OS_fix[0:2],nb.CAL_VAL_OS), OS_fix[2])
        except:
            print("No profile")
            
        self.OD_fix = OD_fix        # one position in each ACT_TIME
        self.OS_fix = OS_fix 
        
    def FeatureExtraction(self):  
        delete = np.where(self.CmdTime['O_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['O_t'] = np.delete(self.CmdTime['O_t'], delete)
        
        delete = np.where(self.CmdTime['CL_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['CL_t'] = np.delete(self.CmdTime['CL_t'], delete)
        
        delete = np.where(self.CmdTime['CR_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['CR_t'] = np.delete(self.CmdTime['CR_t'], delete)
        
        delete = np.where(self.CmdTime['UCR_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['UCR_t'] = np.delete(self.CmdTime['UCR_t'], delete)
            
        if not self.IsVoiceCommand:
            self.NoVoiceCommandFeatureExtraction()
        else:           
            self.VoiceCommandFeatureExtraction()   
    def GetDiagnosis(self):
        OD_fix = self.OD_fix; OS_fix = self.OS_fix
        thr =1.5
        self.nbDx_H = None
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
        for i in range(0,len(nb.ACT_TIME)):
            temp = np.array(self.CmdTime[nb.ACT_TIME[i]])
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
                self.CmdTime[nb.ACT_TIME[i]] = list_temp
            else:
                self.CmdTime[nb.ACT_TIME[i]] = temp
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        FPS = 24
        y_min = -100
        y_max = 100
        time = np.array(range(0,len(OD[0])))/FPS
        time_CL = np.round(self.CmdTime['CL_t']/FPS,2)
        time_CR = np.round(self.CmdTime['CR_t']/FPS,2)
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(nb.EYE)):
            plt.subplot(1,2,i+1)
            if nb.EYE[i] == 'OD':
                plt.vlines(time_CR,y_min,y_max, colors='lightgray', alpha=0.2, linewidth=1)
                x_diff = self.OD_ACT[0,0]-OD[0,:]
                y_diff = self.OD_ACT[0,1]-OD[1,:]
                x_PD = nb.trans_PD(self.AL_OD,x_diff,nb.CAL_VAL_OD)
                y_PD = nb.trans_PD(self.AL_OD,y_diff,nb.CAL_VAL_OD)
            else:
                plt.vlines(time_CL,y_min,y_max, colors='lightgray', alpha=0.2, linewidth=1)
                x_diff = self.OS_ACT[0,0]-OS[0,:]
                y_diff = self.OS_ACT[0,1]-OS[1,:]
                x_PD = nb.trans_PD(self.AL_OS,x_diff,nb.CAL_VAL_OS)
                y_PD = nb.trans_PD(self.AL_OS,y_diff,nb.CAL_VAL_OS)
            
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position (PD)")
            plt.title("Alternate Cover Test "+ nb.EYE[i])
            
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
            plt.ylim([y_min,y_max])
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300) 
    def DrawEyeFig(self):
        ACT = []; OD = self.OD; OS = self.OS
        for i in range(0,len(self.OS_ACT)):
            try:t = np.concatenate(np.array(self.CmdTime[nb.ACT_TIME[i]]))
            except:t = self.CmdTime[nb.ACT_TIME[i]]
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
                #print("Not Detect "+ nb.ACT_TIME[i]) 
        pic_cont = 1
        empt=0
        #fig = plt.figure(figsize=(11.7,8.3))
        fig = plt.gcf()
        fig.set_size_inches(4.2,5, forward=True)
        fig.set_dpi(300)
        for pic in ACT:
            cap = nb.GetVideo(self.csv_path)
            cap.set(1,pic+1)
            ret, im = cap.read()
            j = 1
            while(not ret):
                cap.set(1,pic+1-j)
                ret, im = cap.read()
                j+=1
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
            exec('ax'+str(pic_cont)+ '.set_ylim(int(9*height/10),int(height/8))')
# =============================================================================
#             exec('ax'+str(pic_cont)+ '.set_ylim(int(height),int(0))')
# =============================================================================
            exec('ax'+str(pic_cont)+ '.set_ylabel(nb.ACT_LABEL[pic_cont-1])')
            plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]
        for i in range(0,len(nb.ACT_TIME)):
            if frame_cnt in self.CmdTime[nb.ACT_TIME[i]]:
                text = nb.ACT_STR[i]
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
        