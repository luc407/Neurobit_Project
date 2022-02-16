# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:19:18 2021

@author: luc40
"""
import glob
import os
import math
import sqlite3
import numpy as np
import pandas as pd 
import Neurobit_Lib
import subprocess
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report, CUTReport

main_path = os.getcwd()
Fail = []
Subject = Neurobit_Lib.Neurobit()
            
if __name__== '__main__':    
    csv_files = Subject.GetSubjectFiles(main_path) 
    IsACT_Task = False; IsGaze9_Task = False; IsCUT_Task = False;
    for csv_path in csv_files:
        Subject.GetProfile(csv_path)
        print(Subject.Task)            
        if "9 Gaze" in Subject.Task:
            try: Gaze9_Task.session.append(csv_path)
            except:
                print(csv_path)
                Gaze9_Task = Neurobit_Lib.Gaze9_Task(csv_path)   
                Gaze9_Task.session.append(csv_path)
        elif Subject.Task == "Alternate Cover (ACT)":
            try: ACT_Task.session.append(csv_path)
            except:
                ACT_Task = Neurobit_Lib.ACT_Task(csv_path)  
                ACT_Task.session.append(csv_path)
        elif Subject.Task == "Cover/Uncover (CUT)":
            try: CUT_Task.session.append(csv_path)
            except:
                CUT_Task = Neurobit_Lib.CUT_Task(csv_path)  
                CUT_Task.session.append(csv_path)
        else:
            print("Having no ["+Subject.Task+"] function!")
    
    try: ACT_Task; IsACT_Task = True
    except: print("No ACT_Task!!!")
    try: CUT_Task; IsCUT_Task = True
    except: print("No ACT_Task!!!")
    try: Gaze9_Task; IsGaze9_Task = True
    except: print("No Gaze9_Task!!!")
    """Run Analysis"""
    if IsACT_Task:
        ACT_Task.showVideo = False
        ACT_Task.MergeFile()
        ACT_Task.Exec()        
        
    if IsCUT_Task:
        CUT_Task.showVideo = False
        CUT_Task.MergeFile()
        CUT_Task.Exec()
        
    if IsGaze9_Task:
        Gaze9_Task.showVideo = False
        Gaze9_Task.MergeFile()
        try: Gaze9_Task.Exec(ACT_Task)   
        except: Gaze9_Task.Exec()
        
    """Plot Report"""    
    PDF_Header = sub_head("NeuroSpeed")
    if IsACT_Task:
        Subject_Table   = subject_table(ACT_Task)
        #Clinic_Table    = clinic_table(ACT_Task)
        pdf_path    = os.path.join(ACT_Task.saveReport_path, ACT_Task.FolderName+"_report.pdf")
        pdf         = CreatePDF(pdf_path)
    elif IsCUT_Task:
        Subject_Table   = subject_table(CUT_Task)
        #Clinic_Table    = clinic_table(CUT_Task)
        pdf_path    = os.path.join(CUT_Task.saveReport_path, CUT_Task.FolderName+"_report.pdf")
        pdf         = CreatePDF(pdf_path)    
    elif IsGaze9_Task:
        Subject_Table   = subject_table(Gaze9_Task)
        #Clinic_Table    = clinic_table(Gaze9_Task)
        pdf_path    = os.path.join(Gaze9_Task.saveReport_path, Gaze9_Task.FolderName+"_report.pdf")
        pdf         = CreatePDF(pdf_path)
    
    if IsACT_Task or IsGaze9_Task or IsCUT_Task:
        Sub_Header = sub_head("Clinical relevant data")     
        
        Element = []
        Element.append(PDF_Header)
        Element.append(Subject_Table)
        #Element.append(Sub_Header)
        #Element.append(Clinic_Table)
        try:
            ACTReport(Element, ACT_Task)
        except:
            pass
        Element.append(PageBreak())
        try:
            CUTReport(Element, CUT_Task)
        except:
            pass
        Element.append(PageBreak())
        try:
            Gaze9Report(Element, Gaze9_Task)
        except:
            pass 
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
        cur.execute("""INSERT INTO OcularMotility VALUES(null, '""" + Visit_ID + """', '1', '2', '3', 'ACT_OD_MissingPoint', 'ACT_OS_MissingPoint', 'ACT_Primary', '9Gaze_OD_MissingPoint', '9Gaze_OS_MissingPoint""" +
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
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[0][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[0][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[1][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[1][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[2][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[2][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[3][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[3][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[4][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[4][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[5][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[5][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[6][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[6][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_H[7][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[7][1]) + 
        """', '""" + str(Gaze9_Task.NeurobitDxDev_H[8][1]) + """', '""" + str(Gaze9_Task.NeurobitDxDev_V[8][1]) + """')""")
        con.commit()
