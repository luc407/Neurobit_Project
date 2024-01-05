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
import pandas as pd
import scipy.signal as signal
from scipy.signal import filtfilt
from scipy.signal import find_peaks
from itertools import groupby
# General pre-processing
def fix_blink(data):
    df = pd.DataFrame(data)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    data_f = df.to_numpy().reshape(-1)
    
    return data_f

### SPV Calculation
def isoutlier_pks(locs, pks):
    ## Remove noise peak
    c = 1.4826 # c=-1/(sqrt(2)*erfcinv(3/2))
    MAD = c * np.median(abs(pks - np.median(pks)))  # MAD = c*median(abs(A-median(A)))
    outlier_val = [x for x in pks if (x > 3 * MAD)] # ref function in matlab method "median (default)" https://www.mathworks.com/help/matlab/ref/isoutlier.html#bvlllts-method
    tmp1 = []
    for i in range(len(outlier_val)):
        tmp = np.argwhere(pks == outlier_val[i])
        tmp1 = np.append(tmp1, tmp)
    if (len(tmp1) > 0):
        tmp1 = tmp1.astype(int)
        locs_f = np.delete(locs, tmp1)
        pks_f = np.delete(pks, tmp1)
    else:
        locs_f = np.delete(locs, tmp1)
        pks_f = np.delete(pks, tmp1)
    
    return locs_f, pks_f

## Remove mean outlier
def isoutlier(data):
    outlier_val = [x for x in data if (x > 3 * np.std(data)) or (x < -3 * np.std(data))]
    tmp1 = []
    for i in range(len(outlier_val)):
        tmp = np.argwhere(data == outlier_val[i])
        tmp1 = np.append(tmp1, tmp)
    if (len(tmp1) > 0):
        tmp1 = tmp1.astype(int)
        data_f = np.delete(data, tmp1)
    else:
        data_f = np.delete(data, tmp1)
    
    return data_f

def Nystagmus_extract(data, Fs):
    ## Preprocessing stage
    data_m = data - np.median(data)
    data1 = stats.zscore(data_m)

    ## Non-linear operation
    data2 = np.power(np.diff(data1), 2)

    ## Peak detection
    # saccade last as high as 350 ms / mean is 250 ms
    locs, properties = find_peaks(data2, prominence=(0.1, 5), distance=6) # distance = 250 / (1000/Fs) = 7.5 ; Fs=30sec
    pks = properties.get('prominences')

    return locs, pks

def SPV_computation(data, Interval, medfilt1_para):
    ## Slow phase detection
    data_m = data - np.median(data)
    # true for all elements more than three local scaled MAD from the local median
    c = 1.4826 # c=-1/(sqrt(2)*erfcinv(3/2))
    MAD = c * np.median(abs(np.diff(data_m) - np.median(np.diff(data_m))))  # MAD = c*median(abs(A-median(A)))
    FP_out = np.where(abs(np.diff(data_m)) > (3 * MAD), 0, 1)
    for i in range(1, len(FP_out) - 1):
        if ((FP_out[i-1] & FP_out[i+1]) == 1):
            FP_out[i] = 1
        elif ((FP_out[i-1] | FP_out[i+1]) == 0):
            FP_out[i] = 0
        else:
            FP_out[i] = FP_out[i]
    SP_idx = np.where(FP_out)

    ## Slow Phase Velocity (SPV) parameter
    data_v = np.diff(data_m) / Interval  # for Nystagmus type classification
    #SP_v = signal.medfilt(data_v, medfilt1_para) # for SPV computation
    SP_v = data_v
    SP_v_SP = SP_v[SP_idx]
    SP_v_SP1 = isoutlier(SP_v_SP) # mean remove outlier
    SPV_mean = np.nanmean(SP_v_SP1)
    SPV_std = np.nanstd(SP_v_SP1)
    SPV_med = np.nanmedian(SP_v_SP1)
    if (len(SP_v_SP1) > 0):
        SPV_iqr = np.subtract(*np.percentile(SP_v_SP1, [75, 25]))
    else:
        SPV_iqr = float("nan")

    ## SPV durartion ratio
    # Every VNG waveform (30sec), the duration of slow phase (right or up) over the duration of show phase (left or down)
    # Modified ratio = (short duration / long duration)，high ratio is without Nystagmus
    SPVd_r = np.sum(np.where(SP_v_SP1 > 0, 1, 0))# * Interval
    SPVd_l = np.sum(np.where(SP_v_SP1 < 0, 1, 0))# * Interval
    if (SPVd_r >= SPVd_l):
        SPVd_ratio = SPVd_l / SPVd_r
    else:
        SPVd_ratio = SPVd_r / SPVd_l
    SPV_mean = round(SPV_mean, 2)
    SPV_std = round(SPV_std, 2)
    SPV_med = round(SPV_med, 2)
    SPV_iqr = round(SPV_iqr, 2)
    SPVd_ratio = round(SPVd_ratio, 2)

    return SPV_mean, SPV_std, SPV_med, SPV_iqr, SPVd_ratio, SP_v, SP_idx, data_m, SP_v_SP, SP_v_SP1
    # data_m: zeromean Eye position
    # SP_v: filtered Eye velocity
    # SP_idx: all slow phase index in Eye position and velocity (green dot)
    # SP_v, SP_v_SP, SP_v_SP1, data_v

def Nystagmus_type(data, Interval, locs, data_type):
    ## Nystagmus type classification
    # data_type = 'Horizontal'
    # data_type = 'Vertical'
    data_m = data - np.median(data)
    data_v = np.diff(data_m) / Interval  # for Nystagmus type classification
    saccade_array = np.sign(data_v[locs])
    saccade_num_P = np.sum(np.where(saccade_array == 1, 1, 0))
    saccade_consecnum_P = max([len(list(g)) for i, g in groupby(saccade_array) if i == 1], default = [])
    saccade_num_N = np.sum(np.where(saccade_array == -1, 1, 0))
    saccade_consecnum_N = max([len(list(g)) for i, g in groupby(saccade_array) if i == -1], default = [])
    saccade_num_Z = np.sum(np.where(saccade_array == 0, 1, 0))
    saccade_consecnum_Z = max([len(list(g)) for i, g in groupby(saccade_array) if i == 0], default = [])
    list1 = [saccade_num_P, saccade_num_N, saccade_num_Z]
    saccade_num_max = list1.index(max(list1))
    if saccade_num_max == 0 and (saccade_num_N/saccade_num_P < 0.2):
        saccade_diff = saccade_num_P - saccade_num_N
        saccade_ratio = (saccade_num_P - saccade_num_N) / (saccade_num_P + saccade_num_N)
        if data_type == 'Horizontal':
            type = 'RBN'
        else: # 'Vertical'
            type = 'UBN'
    elif saccade_num_max == 1 and (saccade_num_P/saccade_num_N < 0.2):
        saccade_diff = saccade_num_N - saccade_num_P
        saccade_ratio = (saccade_num_N - saccade_num_P) / (saccade_num_N + saccade_num_P)
        if data_type == 'Horizontal':
            type = 'LBN'
        else: # 'Vertical'
            type = 'DBN'
    elif saccade_num_max == 2:
        saccade_diff = 0
        saccade_ratio = 0
        type = 'Unknown'
    else:
        saccade_diff = 0
        saccade_ratio = 0
        type = 'Jerks'
    saccade_diff = round(saccade_diff, 2)
    saccade_ratio = round(saccade_ratio, 2)

    return type, saccade_diff, saccade_ratio

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
        self.Nystagmus_Quantification()
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
            plt.xlim(0,30)
            if i == 0:
                plt.title('Horizontal')
                plt.scatter(time[self.data['Right']['locs_H']], self.data['Right']['pks_H']+3, marker='v', c='r', alpha=0.5)
                plt.scatter(time[self.data['Left']['locs_H']], self.data['Left']['pks_H']+3, marker='v', c='b', alpha=0.5)
            elif i == 1:
                plt.title('Vertical')
                plt.scatter(time[self.data['Right']['locs_V']], self.data['Right']['pks_V']+3, marker='v', c='r', alpha=0.5)
                plt.scatter(time[self.data['Left']['locs_V']], self.data['Left']['pks_V']+3, marker='v', c='b', alpha=0.5)
            plt.ylim(-30,30)
            plt.xticks(np.arange(0, 31, 5))
            plt.yticks()
        # Place a legend above this subplot, expanding itself to
        # fully use the given bounding box.
        plt.legend(['OD', 'OS', 'OD Nyst', 'OS Nyst'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=3.)
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300, bbox_inches = 'tight')
        plt.close()
        print("DrawEyeTrack")
    def DrawPupil(self):
        self.Preprocessing()
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

    def Nystagmus_Quantification(self):
        data = {'Left':{'Horizontal':{}, 'Vertical':{}, 'locs_H':{}, 'pks_H':{}, 'locs_V':{}, 'pks_V':{}}, 'Right':{'Horizontal':{}, 'Vertical':{}, 'locs_H':{}, 'pks_H':{}, 'locs_V':{}, 'pks_V':{}}, 'Timestamps':{}}
        # eye_select = ['Left', 'Right']
        # dir_select = ['Horizontal', 'Vertical']
        Fs = 30
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/Fs
        data.update({'Timestamps': time})
        for i in range(0,2):
            OD_diff = np.nanmedian(OD[i,:])-OD[i,:]
            OS_diff = np.nanmedian(OS[i,:])-OS[i,:]
            OD_AG = nb.trans_AG(self.AL_OD,OD_diff,nb.CAL_VAL_OD)
            OS_AG = nb.trans_AG(self.AL_OS,OS_diff,nb.CAL_VAL_OS)
            if i == 0:
                data['Right'].update({'Horizontal': OD_AG}) # this Right is for Right eye (at the left side)
                data['Left'].update({'Horizontal': OS_AG}) # this Left is for Left eye (at the right side) 
            else:  
                data['Right'].update({'Vertical': OD_AG})
                data['Left'].update({'Vertical': OS_AG})
        
        self.data = data
        
        ## System parameter setting
        # Predefined video fps
        Interval = 1/Fs
        medfilt1_para = 11 # filter parameter
        T = data['Timestamps'] # load timestamps from data dictionary
        total_time = len(T)/Fs # data time (sec)
        saccade_interval = (T[len(T)-1]/Fs)/10 # num/10s, T[-1]=total frame

        ## output all dictionary data
        saccade_num_dict = {'Left': {}, 'Right':{}}
        saccade_num_FR_dict = {'Left': {}, 'Right':{}}
        SPV_mean_dict = {'Left': {}, 'Right':{}}
        SPV_std_dict = {'Left': {}, 'Right':{}}
        SPV_med_dict = {'Left': {}, 'Right':{}}
        SPV_iqr_dict = {'Left': {}, 'Right':{}}
        SPVd_ratio_dict = {'Left': {}, 'Right':{}}
        data_m_dict = {'Left': {}, 'Right':{}}
        SP_v_dict = {'Left': {}, 'Right':{}}
        SP_idx_dict = {'Left': {}, 'Right':{}}
        type_dict = {'Left': {}, 'Right':{}}
        saccade_diff_dict = {'Left': {}, 'Right':{}}
        saccade_ratio_dict = {'Left': {}, 'Right':{}}
        AutoNyst_dict = {'Left': {}, 'Right':{}}
        SP_v_SP_outlier_filtered_dict = {'Left': {}, 'Right':{}}

        ## Horizontal data / Vertial data as input from Left eye / Right eye
        eye_select = ['Left', 'Right']
        dir_select = ['Horizontal', 'Vertical']
        for eye_key in eye_select:
            for dir_key in dir_select:
                ## VNG data fix zero value
                data_f = fix_blink(data[eye_key][dir_key])
                
                ## Nystagmus trial detection
                locs, pks = Nystagmus_extract(data_f, Fs)
                if dir_key == 'Horizontal':
                    data[eye_key].update({'locs_H': locs})
                    data[eye_key].update({'pks_H': pks})
                else:
                    data[eye_key].update({'locs_V': locs})
                    data[eye_key].update({'pks_V': pks})
                    
                saccade_num = len(locs)
                saccade_num_FR = round(saccade_num / saccade_interval, 2)
                
                ## SPV parameter computation
                SPV_mean, SPV_std, SPV_med, SPV_iqr, SPVd_ratio, SP_v, SP_idx, data_m, SP_v_SP, SP_v_SP1 = SPV_computation(data_f, Interval, medfilt1_para)

                ## Nystagmus type classification
                type, saccade_diff, saccade_ratio = Nystagmus_type(data_f, Interval, locs, dir_key) # data_type use "Horizontal" or "Vertical"
                
                # fix nan to 0
                SPVd_ratio = round(np.nan_to_num(SPVd_ratio),2)
                saccade_ratio = round(np.nan_to_num(saccade_ratio),2)

                # [Ref paper] Winnick, Ariel A., et al. Journal of the Neurological Sciences 442 (2022): 120392.
                # Model 4 : x = − 2.633 + 1.425 (mean SPV) − 3.774 (SP duration ratio) + 0.086 (saccadic difference) + 1.653 (saccadic ratio)
                # AutoNyst  = e^x / (1 - e^x), if AutoNyst >= 0.5 then presence Nyst
                AutoNyst_x = -2.633 + (1.425 * np.abs(SPV_mean)) - (3.774 * SPVd_ratio) + (0.086 * saccade_diff) + (1.653 * saccade_ratio)
                AutoNyst = (np.exp(1) ** AutoNyst_x) / (1 + np.exp(1) ** AutoNyst_x) # sigmoid function to range [0, 1]
                AutoNyst = round(AutoNyst, 2)

                ## Updata dictionary data
                saccade_num_dict[eye_key].update({dir_key: saccade_num})
                saccade_num_FR_dict[eye_key].update({dir_key: saccade_num_FR})
                SPV_mean_dict[eye_key].update({dir_key: SPV_mean})
                SPV_std_dict[eye_key].update({dir_key: SPV_std})
                SPV_med_dict[eye_key].update({dir_key: SPV_med})
                SPV_iqr_dict[eye_key].update({dir_key: SPV_iqr})
                SPVd_ratio_dict[eye_key].update({dir_key: SPVd_ratio})
                data_m_dict[eye_key].update({dir_key: data_m})
                SP_v_dict[eye_key].update({dir_key: SP_v})
                SP_v_SP_outlier_filtered_dict[eye_key].update({dir_key: SP_v_SP1})
                SP_idx_dict[eye_key].update({dir_key: SP_idx})
                type_dict[eye_key].update({dir_key: type})
                saccade_diff_dict[eye_key].update({dir_key: saccade_diff})
                saccade_ratio_dict[eye_key].update({dir_key: saccade_ratio})
                AutoNyst_dict[eye_key].update({dir_key: AutoNyst})

        """Draw Nyst analysis table"""
        self.resultNyst = [['', 'SPV', '', '', '', '', 'Saccadic', '', '', '', '', 'ML Model'],
                        ['', 'Mean', 'Std', 'Median', 'IQR', 'Duration R.', 'Number', 'Firing Rate', 'Type', 'Difference', 'Ratio', 'AutoNyst'],
                        ['OD Hor.(°/s)' , SPV_mean_dict['Right']['Horizontal'], SPV_std_dict['Right']['Horizontal'], SPV_med_dict['Right']['Horizontal'], SPV_iqr_dict['Right']['Horizontal'], SPVd_ratio_dict['Right']['Horizontal'], saccade_num_dict['Right']['Horizontal'], saccade_num_FR_dict['Right']['Horizontal'], type_dict['Right']['Horizontal'], saccade_diff_dict['Right']['Horizontal'], saccade_ratio_dict['Right']['Horizontal'], AutoNyst_dict['Right']['Horizontal']],
                        ['OD Ver.(°/s)'  , SPV_mean_dict['Right']['Vertical'],  SPV_std_dict['Right']['Vertical'],  SPV_med_dict['Right']['Vertical'],  SPV_iqr_dict['Right']['Vertical'],  SPVd_ratio_dict['Right']['Vertical'],  saccade_num_dict['Right']['Vertical'],  saccade_num_FR_dict['Right']['Vertical'],  type_dict['Right']['Vertical'],  saccade_diff_dict['Right']['Vertical'],  saccade_ratio_dict['Right']['Vertical'],  AutoNyst_dict['Right']['Vertical']],
                        ['OS Hor.(°/s)' , SPV_mean_dict['Left']['Horizontal'], SPV_std_dict['Left']['Horizontal'],  SPV_med_dict['Left']['Horizontal'], SPV_iqr_dict['Left']['Horizontal'],  SPVd_ratio_dict['Left']['Horizontal'], saccade_num_dict['Left']['Horizontal'],  saccade_num_FR_dict['Left']['Horizontal'], type_dict['Left']['Horizontal'], saccade_diff_dict['Left']['Horizontal'],  saccade_ratio_dict['Left']['Horizontal'], AutoNyst_dict['Left']['Horizontal']],
                        ['OS Ver.(°/s)'  , SPV_mean_dict['Left']['Vertical'],  SPV_std_dict['Left']['Vertical'],   SPV_med_dict['Left']['Vertical'],  SPV_iqr_dict['Left']['Vertical'],   SPVd_ratio_dict['Left']['Vertical'],  saccade_num_dict['Left']['Vertical'],   saccade_num_FR_dict['Left']['Vertical'],  type_dict['Left']['Vertical'],  saccade_diff_dict['Left']['Vertical'],   saccade_ratio_dict['Left']['Vertical'],  AutoNyst_dict['Left']['Vertical']],
                        ]
