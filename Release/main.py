# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:19:18 2021

@author: luc40
"""
import os
import shutil
import sqlite3
import numpy as np
import pandas as pd 
import tkinter as tk
import subprocess
import Neurobit
from Gaze9_Task import Gaze9_Task
from ACT_Task import ACT_Task
from CUT_Task import CUT_Task
from VF_Task import VF_Task
from Neurobit import Neurobit as NB
from datetime import datetime
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report, CUTReport, VFReport
from calibration import CalibSystem
from reportlab.lib.units import inch

main_path = os.getcwd()
Subject = NB()
        
if __name__== '__main__':    
    csv_files = Subject.GetSubjectFiles(main_path) 
    IsACT_Task      = False; 
    IsGaze9_Task    = False; 
    IsCUT_Task      = False;
    IsVF_Task      = False;
    IsCalibrated    = True;
    for csv_path in csv_files:
        if not IsCalibrated:            
            my_gui = CalibSystem(csv_path)            
            IsCalibrated = True
            Neurobit.OD_WTW = my_gui.OD_WTW
            Neurobit.OS_WTW = my_gui.OS_WTW
            Neurobit.EYE_ORING = [[my_gui.xy[0][0],my_gui.xy[0][1]],
                         [my_gui.xy[3][0],my_gui.xy[3][1]]]
        Subject.GetProfile(csv_path)
        #print(Subject.Task)
        if "9 Gaze Motility Test (9Gaze)" in Subject.Task and Subject.Mode == "OcularMotility":
            try: Gaze9_task.session.append(csv_path)
            except:
                Gaze9_task = Gaze9_Task(csv_path)   
                Gaze9_task.session.append(csv_path)
        elif "Alternate Cover" in Subject.Task and Subject.Mode == "OcularMotility":
            try: ACT_task.session.append(csv_path)
            except:
                ACT_task = ACT_Task(csv_path)  
                ACT_task.session.append(csv_path)
        elif "Cover/Uncover Test (CUT)" in Subject.Task and Subject.Mode == "OcularMotility":
            try: CUT_task.session.append(csv_path)
            except:
                CUT_task = CUT_Task(csv_path)  
                CUT_task.session.append(csv_path)
        elif "Video Frenzel" in Subject.Task and Subject.Mode == "VideoFrenzel":
            try: 
                VF_task.GetProfile(csv_path)
                VF_task.session.append(csv_path)
            except:
                VF_task = VF_Task(csv_path)  
                VF_task.GetProfile(csv_path)
                VF_task.session.append(csv_path)
        else:
            pass
    
    try: ACT_task; IsACT_Task = True
    except: pass
    try: CUT_task; IsCUT_Task = True
    except: pass
    try: Gaze9_task; IsGaze9_Task = True
    except: pass
    try: VF_task; IsVF_Task = True
    except: pass
    """Run Analysis"""
    if IsACT_Task:
        ACT_task.showVideo = False
        ACT_task.MergeFile()
        ACT_task.Exec()
    else:
        ACT_task = ACT_Task(csv_path)  
        ACT_task.miss_OD = np.nan
        ACT_task.miss_OS = np.nan
        ACT_task.NeurobitDx_H = np.nan
        ACT_task.NeurobitDx_V = np.nan
        ACT_task.NeurobitDxDev_H = np.nan
        ACT_task.NeurobitDxDev_V = np.nan
        
    if IsCUT_Task:
        CUT_task.showVideo = False
        CUT_task.MergeFile()
        CUT_task.Exec()
    else:
        CUT_task = CUT_Task(csv_path)  
        CUT_task.miss_OD = np.nan
        CUT_task.miss_OS = np.nan
        CUT_task.NeurobitDx_H = np.nan
        CUT_task.NeurobitDx_V = np.nan
        CUT_task.NeurobitDxDev_H = np.nan
        CUT_task.NeurobitDxDev_V = np.nan        
        
    if IsGaze9_Task:
        Gaze9_task.showVideo = False
        Gaze9_task.MergeFile()
        if IsACT_Task: Gaze9_task.Exec(ACT_task)   
        else: Gaze9_task.Exec()
    else:
        Gaze9_task = Gaze9_Task(csv_path)  
        Gaze9_task.miss_OD = np.nan
        Gaze9_task.miss_OS = np.nan
        Gaze9_task.NeurobitDxDev_H = np.empty([9,2])*np.nan
        Gaze9_task.NeurobitDxDev_V = np.empty([9,2])*np.nan
        Gaze9_task.Diff_H = np.empty([9,2])*np.nan
        Gaze9_task.Diff_V = np.empty([9,2])*np.nan
     
    if IsVF_Task:
        VF_task.showVideo = False
        VF_task.MergeFile()
        VF_task.Exec()
    else:
        VF_task = VF_Task(csv_path)  

    """Plot OcularMotility Report"""    
    PDF_Header = sub_head("NeuroSpeed")
    if IsACT_Task:
        Subject_Table   = subject_table(ACT_task)
        pdf_path    = os.path.join(ACT_task.saveReport_path,
                                   datetime.now().strftime("%Y%m%d")+"_"+datetime.now().strftime("%H%M%S")+"_"+
                                   ACT_task.FolderName.split("_")[-1]+"_OcularMotility.pdf")
        pdf         = CreatePDF(pdf_path)
    elif IsCUT_Task:
        Subject_Table   = subject_table(CUT_task)
        pdf_path    = os.path.join(CUT_task.saveReport_path,
                                   datetime.now().strftime("%Y%m%d")+"_"+datetime.now().strftime("%H%M%S")+"_"+
                                   CUT_task.FolderName.split("_")[-1]+"_OcularMotility.pdf")
        pdf         = CreatePDF(pdf_path)    
    elif IsGaze9_Task:
        Subject_Table   = subject_table(Gaze9_task)
        pdf_path    = os.path.join(Gaze9_task.saveReport_path,
                                   datetime.now().strftime("%Y%m%d")+"_"+datetime.now().strftime("%H%M%S")+"_"+
                                   Gaze9_task.FolderName.split("_")[-1]+"_OcularMotility.pdf")
        pdf         = CreatePDF(pdf_path)
        
    if IsACT_Task or IsGaze9_Task or IsCUT_Task:
        Sub_Header = sub_head("Clinical relevant data")     
        
        Element = []
        Element.append(PDF_Header)
        Element.append(Subject_Table)
        #Element.append(Sub_Header)
        #Element.append(Clinic_Table)
        if IsACT_Task:
            ACTReport(Element, ACT_task)

        Element.append(PageBreak())
        if IsCUT_Task:
            CUTReport(Element, CUT_task)
            Element.append(PageBreak())

        if IsGaze9_Task:
            Gaze9Report(Element, Gaze9_task)

        pdf.build(Element)
        subprocess.Popen(pdf_path, shell=True)

        """Update NeurobitNS01-1.db for Ocular Motility"""
        con = sqlite3.connect(os.path.join(main_path, 'NeurobitNS01-1.db'))
        cur = con.cursor()
        lastconnection = datetime.strptime(Subject.Date, "%Y%m%d").strftime('%Y/%m/%d')
        cur.execute("SELECT [Procedure_ID] FROM Visit WHERE [Datetime]='" + lastconnection + "' AND [Patient_ID]='" + Subject.ID + "' AND [Examiner_ID]='" + Subject.Doctor + "' AND [Procedure]='OcularMotility'")
        procedure_ID = str(cur.fetchall()[-1][0]+1)
        cur.execute("INSERT INTO Visit VALUES(null, '" + lastconnection + "', '" + Subject.ID + "', '" + Subject.Doctor + "', 'OcularMotility', '" + procedure_ID + "', null)")
        cur.execute("SELECT [ID] FROM Visit WHERE [Datetime]='" + lastconnection + "' AND [Patient_ID]='" + Subject.ID + "' AND [Examiner_ID]='" + Subject.Doctor + "' AND [Procedure]='OcularMotility' AND [procedure_ID]='" + procedure_ID + "'")
        Visit_ID = str(cur.fetchall()[-1][0])
        cur.execute("""INSERT INTO OcularMotility VALUES(null, '""" + Visit_ID + """', null, null, null, null""" +
        """, '""" + str(ACT_task.miss_OD) + """', '""" + str(ACT_task.miss_OS) + """', '""" + str(ACT_task.NeurobitDx_H) + """', '""" + str(ACT_task.NeurobitDx_V) + """', '""" + str(ACT_task.NeurobitDxDev_H) + """', '""" + str(ACT_task.NeurobitDxDev_V) + 
        """', '""" + str(CUT_task.miss_OD) + """', '""" + str(CUT_task.miss_OS) + """', '""" + str(CUT_task.NeurobitDx_H) + """', '""" + str(CUT_task.NeurobitDx_V) + """', '""" + str(CUT_task.NeurobitDxDev_H) + """', '""" + str(CUT_task.NeurobitDxDev_V) + 
        """', '""" + str(Gaze9_task.miss_OD) + """', '""" + str(Gaze9_task.miss_OS) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[0][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[0][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[1][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[1][0]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[2][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[2][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[3][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[3][0]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[4][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[4][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[5][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[5][0]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[6][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[6][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[7][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[7][0]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[8][0]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[8][0]) +
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[0][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[0][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[1][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[1][1]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[2][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[2][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[3][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[3][1]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[4][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[4][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[5][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[5][1]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[6][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[6][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_H[7][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[7][1]) + 
        """', '""" + str(Gaze9_task.NeurobitDxDev_H[8][1]) + """', '""" + str(Gaze9_task.NeurobitDxDev_V[8][1]) +
        """', '""" + str(np.round(Gaze9_task.Diff_H[0],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[0],1)) + """', '""" + str(np.round(Gaze9_task.Diff_H[1],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[1],1)) + 
        """', '""" + str(np.round(Gaze9_task.Diff_H[2],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[2],1)) + """', '""" + str(np.round(Gaze9_task.Diff_H[3],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[3],1)) + 
        """', '""" + str(np.round(Gaze9_task.Diff_H[4],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[4],1)) + """', '""" + str(np.round(Gaze9_task.Diff_H[5],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[5],1)) + 
        """', '""" + str(np.round(Gaze9_task.Diff_H[6],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[6],1)) + """', '""" + str(np.round(Gaze9_task.Diff_H[7],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[7],1)) + 
        """', '""" + str(np.round(Gaze9_task.Diff_H[8],1)) + """', '""" + str(np.round(Gaze9_task.Diff_V[8],1)) + """')""")
        con.commit()
    
    """Plot VideoFrenzel Report"""  
    PDF_Header = sub_head("NeuroSpeed")
    if IsVF_Task:
        Subject_Table   = subject_table(VF_task)
        pdf_path    = os.path.join(VF_task.saveReport_path,
                                   VF_task.FileName+".pdf")
        pdf         = CreatePDF(pdf_path)                
        Element = []
        Element.append(PDF_Header)
        Element.append(Spacer(1, inch * 0.10))
        Element.append(Subject_Table)
        Element.append(Spacer(1, inch * 0.10))
        VFReport(Element, VF_task)
        pdf.build(Element)
        subprocess.Popen(pdf_path, shell=True)

        """Update NeurobitNS01-1.db for VideoFrenzel"""
        con = sqlite3.connect(os.path.join(main_path, 'NeurobitNS01-1.db'))
        cur = con.cursor()
        lastconnection = datetime.strptime(VF_task.Date, "%Y%m%d").strftime('%Y/%m/%d')
        cur.execute("SELECT [Procedure_ID] FROM Visit WHERE [Datetime]='" + lastconnection + "' AND [Patient_ID]='" + VF_task.ID + "' AND [Examiner_ID]='" + VF_task.Doctor + "' AND [Procedure]='VideoFrenzel'")
        procedure_ID = str(cur.fetchall()[-1][0]+1)
        cur.execute("INSERT INTO Visit VALUES(null, '" + lastconnection + "', '" + VF_task.ID + "', '" + VF_task.Doctor + "', 'VideoFrenzel', '" + procedure_ID + "', null)")
        cur.execute("SELECT [ID] FROM Visit WHERE [Datetime]='" + lastconnection + "' AND [Patient_ID]='" + VF_task.ID + "' AND [Examiner_ID]='" + VF_task.Doctor + "' AND [Procedure]='VideoFrenzel' AND [Procedure_ID]='" + procedure_ID + "'")
        Visit_ID = str(cur.fetchall()[-1][0])
        cur.execute("""INSERT INTO VideoFrenzel VALUES(null, '""" + Visit_ID + 
        """', '""" + str(VF_task.result['Mean']['Right']) + """', '""" + str(VF_task.result['Min']['Right']) + """', '""" + str(VF_task.result['Min']['Right']) + """', '""" + str(VF_task.result['Std']['Right']) + 
        """', '""" + str(VF_task.result['Mean']['Left']) + """', '""" + str(VF_task.result['Min']['Left']) + """', '""" + str(VF_task.result['Min']['Left']) + """', '""" + str(VF_task.result['Std']['Left']) + 
        """', '""" + str(VF_task.result['Mean']['Diff_label']) + """', '""" + str(VF_task.result['Min']['Diff_label']) + """', '""" + str(VF_task.result['Min']['Diff_label']) + """', '""" + str(VF_task.result['Std']['Diff_label']) + 
        """', '""" + str(VF_task.result['Mean']['Diff']) + """', '""" + str(VF_task.result['Min']['Diff']) + """', '""" + str(VF_task.result['Min']['Diff']) + """', '""" + str(VF_task.result['Std']['Diff']) + """')""")
        con.commit()     


    shutil.rmtree(Subject.save_path)


