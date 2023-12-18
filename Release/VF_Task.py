# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 10:23:22 2022

@author: luc40
"""

import os
import cv2
import numpy as np
import Neurobit as nb
from Neurobit import Neurobit
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

class VF_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "VF"
        self.Mode = "VideoFrenzel"
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
        self.GetEyePosition()
        self.FeatureExtraction()
        
        self.DrawEyeTrack()
        self.DrawPupil()            
    
    def FeatureExtraction(self):
        pupil_size_OD_All = self.OD[2,:]*nb.CAL_VAL_OD*2
        pupil_size_OS_All = self.OS[2,:]*nb.CAL_VAL_OS*2
        self.result = {'Mean': {}, 'Min': {}, 'Max': {}, 'Std': {}}

        ## Find parameters
        # Size (Mean)
        list1 = [round(np.nanmean(pupil_size_OD_All), 2), round(np.nanmean(pupil_size_OS_All), 2)]
        index = list1.index(max(list1))
        size_diff_value = round(abs(round(np.nanmean(pupil_size_OD_All), 2) - round(np.nanmean(pupil_size_OS_All), 2)), 2)
        if (size_diff_value == 0):
            size_diff = 'OD = OS'
        else:
            if (index):
                size_diff = 'OS > OD'
            else:
                size_diff = 'OD > OS'
        
        
        # Min
        list1 = [round(np.nanmin(pupil_size_OD_All), 2), round(np.nanmin(pupil_size_OS_All), 2)]
        index = list1.index(max(list1))
        min_diff_value = round(abs(round(np.nanmin(pupil_size_OD_All), 2) - round(np.nanmin(pupil_size_OS_All), 2)), 2)
        if (min_diff_value == 0):
            min_diff = 'OD = OS'
        else:
            if (index):
                min_diff = 'OS > OD'
            else:
                min_diff = 'OD > OS'
        
        # Max
        list1 = [round(np.nanmax(pupil_size_OD_All), 2), round(np.nanmax(pupil_size_OS_All), 2)]
        index = list1.index(max(list1))
        max_diff_value = round(abs(round(np.nanmax(pupil_size_OD_All), 2) - round(np.nanmax(pupil_size_OS_All), 2)), 2)
        if (max_diff_value == 0):
            max_diff = 'OD = OS'
        else:
            if (index):
                max_diff = 'OS > OD'
            else:
                max_diff = 'OD > OS'
        
        
        # Std
        list1 = [round(np.nanstd(pupil_size_OD_All), 2), round(np.nanstd(pupil_size_OS_All), 2)]
        index = list1.index(max(list1))
        std_diff_value = round(abs(round(np.nanstd(pupil_size_OD_All), 2) - round(np.nanstd(pupil_size_OS_All), 2)), 2)
        if (std_diff_value == 0):
            std_diff = 'OD = OS'
        else:
            if (index):
                std_diff = 'OS > OD'
            else:
                std_diff = 'OD > OS'
        
        self.result['Mean'].update({'Right': round(np.nanmean(pupil_size_OD_All), 2), 'Left': round(np.nanmean(pupil_size_OS_All), 2), 'Diff_label': size_diff, 'Diff': size_diff_value})
        self.result['Min'].update( {'Right': round(np.nanmin(pupil_size_OD_All), 2),  'Left': round(np.nanmin(pupil_size_OS_All), 2),  'Diff_label': min_diff, 'Diff': min_diff_value})
        self.result['Max'].update( {'Right': round(np.nanmax(pupil_size_OD_All), 2),  'Left': round(np.nanmax(pupil_size_OS_All), 2),  'Diff_label': max_diff, 'Diff': max_diff_value})
        self.result['Std'].update( {'Right': round(np.nanstd(pupil_size_OD_All), 2),  'Left': round(np.nanstd(pupil_size_OS_All), 2),  'Diff_label': std_diff, 'Diff': std_diff_value})
        
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/30
        plt.rcParams["figure.figsize"] = (nb.TABLE_WIDTH*1.5, nb.TABLE_WIDTH *3/5)
        plt.rcParams["font.family"] = "Arial"           
        for i in range(0,2):
            OD_diff = np.nanmedian(OD[i,:])-OD[i,:]
            OS_diff = np.nanmedian(OS[i,:])-OS[i,:]
            OD_AG = nb.trans_AG(self.AL_OD,OD_diff,nb.CAL_VAL_OD)
            OS_AG = nb.trans_AG(self.AL_OS,OS_diff,nb.CAL_VAL_OS)
            plt.subplot(2,1,i+1)
            ax = plt.gca()
             
            ax.grid(which='major', linestyle='dotted', linewidth=0.3)
            ax.grid(which='minor', linestyle='dotted', linewidth=0.3)
            majorLocator = MultipleLocator(5)
            minorLocator = MultipleLocator(1)
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_minor_locator(minorLocator)
            majorLocator = MultipleLocator(10)
            minorLocator = MultipleLocator(5)
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_minor_locator(minorLocator)
            # ax.axes.yaxis.set_ticklabels([])
            plt.plot(time,OD_AG, nb.line_color_palatte['reds'][2], linewidth=0.5)
            plt.plot(time,OS_AG, nb.line_color_palatte['blues'][2], linewidth=0.5)  
            plt.ylabel('Position (°)')
            plt.xlabel('Time (s)')
            plt.legend(['OD', 'OS'])
            #plt.xlim(0,30)
            if i == 0:
                plt.title('Horizontal')
            elif i == 1:
                plt.title('Vertical')
            plt.ylim(np.nanmin(OD_AG)-5,np.nanmax(OD_AG)+5)
            plt.xticks(np.arange(0, 31, 5))
            plt.yticks()
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300, bbox_inches = 'tight')
        plt.close()
        print("DrawEyeTrack")
    def DrawPupil(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/30
        plt.rcParams["figure.figsize"] = (nb.TABLE_WIDTH*1.5, nb.TABLE_WIDTH *3/10)
        plt.rcParams["font.family"] = "Arial"           
        plt.plot(time,OD[2,:]*nb.CAL_VAL_OD*2, nb.line_color_palatte['reds'][2], linewidth=0.5)
        plt.plot(time,OS[2,:]*nb.CAL_VAL_OS*2, nb.line_color_palatte['blues'][2], linewidth=0.5)            
        plt.xlim(0,30)
        plt.ylim(2,12)
        plt.title('Pupil Size Timeseries plot')
        plt.ylabel('Diameter (mm)')
        plt.xlabel('Time (s)')
        plt.legend(['OD', 'OS'], fontsize=8)
        plt.savefig(os.path.join(self.saveImage_path,"DrawPupil.png"), dpi=300, bbox_inches = 'tight')
        plt.close()
        print("DrawPupil")
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]
        
