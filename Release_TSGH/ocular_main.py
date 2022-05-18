# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:19:18 2021

@author: luc40
"""
from cmath import nan
import glob
import os
import math
import sqlite3
import numpy as np
import pandas as pd 
import Neurobit_Lib
import subprocess
import shutil
import tkinter as tk
import Neurobit_Lib
from datetime import datetime
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report, CUTReport
from calibration import CalibSystem



main_path = os.getcwd()
Fail = []
Subject = Neurobit_Lib.Neurobit()
            
if __name__== '__main__':    
    csv_files = Subject.GetSubjectFiles(main_path) 
    IsACT_Task = False; IsGaze9_Task = False; IsCUT_Task = False; IsCalibrated = False
    for csv_path in csv_files:
        if not IsCalibrated:
            root = tk.Tk()
            my_gui = CalibSystem(root, csv_path)
            root.mainloop()
            IsCalibrated = True
            Neurobit_Lib.OD_WTW = my_gui.OD_WTW
            Neurobit_Lib.OS_WTW = my_gui.OS_WTW
        Subject.GetProfile(csv_path)
        
        if "9 Gaze Motility Test (9Gaze)" in Subject.Task:
            try: Gaze9_Task.session.append(csv_path)
            except:
                Gaze9_Task = Neurobit_Lib.Gaze9_Task(csv_path)   
                Gaze9_Task.session.append(csv_path)
        elif "Alternate Cover" in Subject.Task:
            try: ACT_Task.session.append(csv_path)
            except:
                ACT_Task = Neurobit_Lib.ACT_Task(csv_path)  
                ACT_Task.session.append(csv_path)
        elif "Cover/Uncover Test (CUT)" in Subject.Task:
            try: CUT_Task.session.append(csv_path)
            except:
                CUT_Task = Neurobit_Lib.CUT_Task(csv_path)  
                CUT_Task.session.append(csv_path)
        else:
            pass
    
    try: ACT_Task; IsACT_Task = True
    except: pass#print("No ACT_Task!!!")
    try: CUT_Task; IsCUT_Task = True
    except: pass#print("No ACT_Task!!!")
    try: Gaze9_Task; IsGaze9_Task = True
    except: pass#print("No Gaze9_Task!!!")
    """Run Analysis"""
    if IsACT_Task:
        ACT_Task.showVideo = False
        ACT_Task.MergeFile()
        ACT_Task.Exec()
    else:
        ACT_Task = Neurobit_Lib.ACT_Task(csv_path)  
        ACT_Task.miss_OD = np.nan
        ACT_Task.miss_OS = np.nan
        ACT_Task.NeurobitDx_H = np.nan
        ACT_Task.NeurobitDx_V = np.nan
        ACT_Task.NeurobitDxDev_H = np.nan
        ACT_Task.NeurobitDxDev_V = np.nan
        
    if IsCUT_Task:
        CUT_Task.showVideo = False
        CUT_Task.MergeFile()
        CUT_Task.Exec()
    else:
        CUT_Task = Neurobit_Lib.CUT_Task(csv_path)  
        CUT_Task.miss_OD = np.nan
        CUT_Task.miss_OS = np.nan
        CUT_Task.NeurobitDx_H = np.nan
        CUT_Task.NeurobitDx_V = np.nan
        CUT_Task.NeurobitDxDev_H = np.nan
        CUT_Task.NeurobitDxDev_V = np.nan
        
    if IsGaze9_Task:
        Gaze9_Task.showVideo = False
        Gaze9_Task.MergeFile()
        if IsACT_Task: Gaze9_Task.Exec(ACT_Task)   
        else: Gaze9_Task.Exec()
    else:
        Gaze9_Task = Neurobit_Lib.Gaze9_Task(csv_path)  
        Gaze9_Task.miss_OD = np.nan
        Gaze9_Task.miss_OS = np.nan
        Gaze9_Task.NeurobitDxDev_H = np.empty([9,2])*np.nan
        Gaze9_Task.NeurobitDxDev_V = np.empty([9,2])*np.nan
        Gaze9_Task.Diff_H = np.empty([9,2])*np.nan
        Gaze9_Task.Diff_V = np.empty([9,2])*np.nan
        
    """Plot Report"""    
    PDF_Header = sub_head("NeuroSpeed")
    if IsACT_Task:
        Subject_Table   = subject_table(ACT_Task)
        Clinic_Table    = clinic_table(ACT_Task)
        pdf_path    = os.path.join(ACT_Task.saveReport_path,
                                   ACT_Task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
                                   "_OcularMotility.pdf")
        pdf         = CreatePDF(pdf_path)
    elif IsCUT_Task:
        Subject_Table   = subject_table(CUT_Task)
        Clinic_Table    = clinic_table(CUT_Task)
        pdf_path    = os.path.join(CUT_Task.saveReport_path, 
                                   CUT_Task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
                                   "_OcularMotility.pdf")
        pdf         = CreatePDF(pdf_path)    
    elif IsGaze9_Task:
        Subject_Table   = subject_table(Gaze9_Task)
        Clinic_Table    = clinic_table(Gaze9_Task)
        pdf_path    = os.path.join(Gaze9_Task.saveReport_path, 
                                   Gaze9_Task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
                                   "_OcularMotility.pdf")
        pdf         = CreatePDF(pdf_path)
    
    if IsACT_Task or IsGaze9_Task or IsCUT_Task:
        Sub_Header = sub_head("Clinical relevant data")     
        
        Element = []
        Element.append(PDF_Header)
        Element.append(Subject_Table)
        Element.append(Sub_Header)
        Element.append(Clinic_Table)
        if IsACT_Task:
            ACTReport(Element, ACT_Task)

        Element.append(PageBreak())
        if IsCUT_Task:
            CUTReport(Element, CUT_Task)
            Element.append(PageBreak())

        if IsGaze9_Task:
            Gaze9Report(Element, Gaze9_Task)

        pdf.build(Element)
        subprocess.Popen(pdf_path, shell=True)

        con = sqlite3.connect(os.path.join(main_path, 'NeurobitNS01-1.db'))
        cur = con.cursor()
        lastconnection = datetime.strptime(Subject.Date, "%Y%m%d").strftime('%Y/%m/%d')
        cur.execute("SELECT [Procedure_ID] FROM Visit WHERE [Datetime]='" + lastconnection + "' AND [Patient_ID]='" + Subject.ID + "' AND [Examiner_ID]='" + Subject.Doctor + "' AND [Procedure]='OcularMotility'")
        procedure_ID = str(cur.fetchall()[-1][0]+1)
        cur.execute("INSERT INTO Visit VALUES(null, '" + lastconnection + "', '" + Subject.ID + "', '" + Subject.Doctor + "', 'OcularMotility', '" + procedure_ID + "', null)")
        cur.execute("SELECT [ID] FROM Visit WHERE [Datetime]='" + lastconnection + "' AND [Patient_ID]='" + Subject.ID + "' AND [Examiner_ID]='" + Subject.Doctor + "' AND [Procedure]='OcularMotility' AND [procedure_ID]='" + procedure_ID + "'")
        Visit_ID = str(cur.fetchall()[-1][0])
        cur.execute("""INSERT INTO OcularMotility VALUES(null, '""" + Visit_ID + """', null, null, null, null""" +
        """, '""" + str(ACT_Task.miss_OD) + """', '""" + str(ACT_Task.miss_OS) + """', '""" + str(ACT_Task.NeurobitDx_H) + """', '""" + str(ACT_Task.NeurobitDx_V) + """', '""" + str(ACT_Task.NeurobitDxDev_H) + """', '""" + str(ACT_Task.NeurobitDxDev_V) + 
        """', '""" + str(CUT_Task.miss_OD) + """', '""" + str(CUT_Task.miss_OS) + """', '""" + str(CUT_Task.NeurobitDx_H) + """', '""" + str(CUT_Task.NeurobitDx_V) + """', '""" + str(CUT_Task.NeurobitDxDev_H) + """', '""" + str(CUT_Task.NeurobitDxDev_V) + 
        """', '""" + str(Gaze9_Task.miss_OD) + """', '""" + str(Gaze9_Task.miss_OS) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[0][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[0][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[1][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[1][0]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[2][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[2][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[3][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[3][0]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[4][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[4][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[5][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[5][0]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[6][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[6][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[7][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[7][0]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[8][0]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[8][0]) +
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[0][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[0][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[1][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[1][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[2][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[2][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[3][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[3][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[4][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[4][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[5][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[5][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[6][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[6][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[7][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[7][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[8][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[8][1]) +
        """', '""" + str(np.round(Gaze9_Task.Diff_H[0],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[0],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_H[1],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[1],1)) + 
        """', '""" + str(np.round(Gaze9_Task.Diff_H[2],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[2],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_H[3],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[3],1)) + 
        """', '""" + str(np.round(Gaze9_Task.Diff_H[4],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[4],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_H[5],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[5],1)) + 
        """', '""" + str(np.round(Gaze9_Task.Diff_H[6],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[6],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_H[7],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[7],1)) + 
        """', '""" + str(np.round(Gaze9_Task.Diff_H[8],1)) + """', '""" + str(np.round(Gaze9_Task.Diff_V[8],1)) + """')""")
        con.commit()
        
        try: shutil.rmtree(ACT_Task.save_MainPath+"\\"+ACT_Task.task)
        except: pass
        try: shutil.rmtree(CUT_Task.save_MainPath+"\\"+CUT_Task.task)
        except: pass
        try: shutil.rmtree(Gaze9_Task.save_MainPath+"\\"+Gaze9_Task.task)
        except: pass
