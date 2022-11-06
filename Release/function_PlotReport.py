# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:32:28 2021

@author: luc40
"""

# ReportLab imports
import os
import numpy as np
import Neurobit as nb
from PIL import Image as pImage
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer 
from reportlab.graphics import renderPDF, renderPM

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial_bold', 'arialbd.ttf'))
pdfmetrics.registerFont(TTFont('TSans', 'TaipeiSansTCBeta-Regular.ttf'))
pdfmetrics.registerFont(TTFont('TSans_bold', 'TaipeiSansTCBeta-Bold.ttf'))

def main_head(headtext):
    Style=getSampleStyleSheet()
    bt = Style['Normal']    #字體的樣式
    bt.fontName='Helvetica-Bold'     #使用的字體
    bt.fontSize=18          #字號
    bt.wordWrap = 'Normal'     #該屬性支持自動換行，'CJK'是中文模式換行，用於英文中會截斷單詞造成閱讀困難，可改爲'Normal'
    bt.spaceAfter= 16
    return Paragraph(headtext,bt)

def sub_head(headtext):
    Style=getSampleStyleSheet()
    bt = Style['Normal']    #字體的樣式
    bt.fontName='Helvetica-Bold'     #使用的字體
    bt.fontSize=14          #字號
    bt.wordWrap = 'Normal'     #該屬性支持自動換行，'CJK'是中文模式換行，用於英文中會截斷單詞造成閱讀困難，可改爲'Normal'
    bt.spaceBefore= 10
    bt.spaceAfter= 10
    return Paragraph(headtext,bt)

def con_text(headtext):
    Style=getSampleStyleSheet()
    bt = Style['Normal']    #字體的樣式
    bt.fontName='Helvetica-Bold'     #使用的字體
    bt.fontSize=10         #字號
    bt.wordWrap = 'Normal'     #該屬性支持自動換行，'CJK'是中文模式換行，用於英文中會截斷單詞造成閱讀困難，可改爲'Normal'
    bt.spaceBefore= 10
    bt.spaceAfter= 10
    return Paragraph(headtext,bt)
    
def subject_table(Subject):
    if Subject.ID:
        data = [['Patient ID: ' + Subject.ID,       'Date of Birth: ' + Subject.DoB,    'Exam Date: ' + Subject.Date,   'Examiner ID: ' + Subject.Doctor],
                ['Patient Name: ' + Subject.Name,   'Sex: ' + Subject.Gender,        'Age: ' + Subject.Age,          'Height: ' + str(Subject.Height)]
        ]   
    else:
        data = [['Patient ID: ' + Subject.ID,       'Date of Birth: ' + str(Subject.Profile_ind),    'Exam Date: ' + Subject.Date,   'Doctor: ' + Subject.Doctor],
                ['Patient Name: ' + str(Subject.Profile_ind),   'Gender: ' + str(Subject.Profile_ind),        'Age: ' + str(Subject.Profile_ind),          'Height: ' + str(Subject.Profile_ind)]
        ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),    # 字體
        ('FONTNAME', (0, 1), (0, 1), 'TSans'),    # 字體
        ('FONTSIZE', (0, 0), (-1, 0), 10),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),        # 對齊
        ('VALIGN', (-1, 0), (-2, 0), 'MIDDLE'),     # 對齊
        
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),   # 設置表格框線爲grey色，線寬爲0.5
    ]    
    colWidths = (nb.TABLE_WIDTH / len(data[0])) * inch     # 每列的寬度
    component_table = Table(dis_list,colWidths = colWidths, style=style)    
    return component_table

def clinic_table(Subject):
    if Subject.ID:
        data = [['Hx: ' + Subject.Dx ,'','','','','','','','','',''],
                ['',    'VAsc',          'VAcc',        'Auto-Ref',     'pupil',            'WTW',          'AXL',          'Hertel',           '',                 'PD',       'Stereo'],
                ['OD',  Subject.VA_OD, Subject.BCVA_OD, Subject.Ref_OD,  Subject.pupil_OD,  Subject.WTW_OD, Subject.AL_OD, Subject.Hertal_OD,  Subject.Hertal_Len, Subject.PD, Subject.Stereo],
                ['OS',  Subject.VA_OS, Subject.BCVA_OS, Subject.Ref_OS,  Subject.pupil_OS,  Subject.WTW_OS, Subject.AL_OS, Subject.Hertal_OS,  '',                 '',         ''],
        ]     
    else:
        data = [['Hx: ' + str(Subject.Profile_ind),'','','','','','','','','',''],
                ['',    'VAsc',          'VAcc',        'Auto-Ref',     'pupil',            'WTW',          'AXL',          'Hertel',           '',                 'PD',       'Stereo'],
                ['OD',  '', '', '', '', '', '', '', '', '', ''],
                ['OS',  '', '', '', '', '', '', '', '', '', ''],
        ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),    # 字體
        ('FONTSIZE', (0, 0), (-1, 0), 10),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, 0), 'LEFT'),        # 對齊
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('SPAN',(0,0),(-1,0)),
        ('SPAN',(7,1),(8,1)),
        ('SPAN',(8,2),(8,3)),
        ('SPAN',(9,2),(9,3)),
        ('SPAN',(10,2),(10,3)),
        
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),   # 設置表格框線爲grey色，線寬爲0.5
    ]    
    colWidths = (nb.TABLE_WIDTH / len(data[0])) * inch     # 每列的寬度
    component_table = Table(dis_list,colWidths = colWidths, style=style)    
    return component_table
def diagnose_table(OLD_ACT_Task):
    data=[['','','','','',''],
          ['','','','','',''],
          ['','',OLD_ACT_Task.NeurobitDx_H,'\n'+str(np.round(OLD_ACT_Task.NeurobitDxDev_H,1))+' PD','',''],
          ['','',OLD_ACT_Task.NeurobitDx_V,'\n'+str(np.round(OLD_ACT_Task.NeurobitDxDev_V,1))+' PD','',''],
          ['','','','','',''],
          ['','','','','','']]   
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),    # 字體
        ('FONTSIZE', (0, 0), (-1, -1), 10),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('BOX', (0, 0), (1, 1), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (0, 2), (1, 3), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (0, 4), (1, 5), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (2, 0), (3, 1), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (2, 2), (3, 3), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (2, 4), (3, 5), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (4, 0), (5, 1), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (4, 2), (5, 3), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
        ('BOX', (4, 4), (5, 5), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
    ]    
    colWidths = (nb.GAZE_TABLE_WIDTH / len(data[0])) * inch     # 每列的寬度
    component_table = Table(dis_list, 
                            colWidths=colWidths, 
                            rowHeights=colWidths/2,
                            style=style)    
    return component_table
def quality_bar(OD, OS, Task):
    miss_OD = round(np.count_nonzero(np.isnan(OD[0]))/len(OD[0]),2)
    miss_OS = round(np.count_nonzero(np.isnan(OS[0]))/len(OS[0]),2)
    Task.miss_OD = miss_OD
    Task.miss_OS = miss_OS    

    data=[['Missing Point: ','OD: ', str(int(miss_OD*100))+'%','OS: ', str(int(miss_OS*100))+'%']
          ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    
    if 1-miss_OD>=0.9:
        OD_color = colors.lime
    elif 1-miss_OD>=0.7:
        OD_color = colors.yellow
    else:
        OD_color = colors.red
    if 1-miss_OS>=0.9:
        OS_color = colors.lime
    elif 1-miss_OS>=0.7:
        OS_color = colors.yellow
    else:
        OS_color = colors.red
        
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),    # 字體
        ('FONTSIZE', (0, 0), (-1, -1), 8),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('BOX', (0, 0), (-1, -1), 1, colors.grey),   # 設置表格框線爲grey色，線寬爲1
        ('BACKGROUND', (2, 0), (2, 0), OD_color),
        ('BACKGROUND', (4, 0), (4, 0), OS_color)
    ]    
    component_table = Table(dis_list, style=style, rowHeights=10)    
    return component_table
def EyeTrackImage(file_path):
    im = Image(file_path, width=7.2 *inch, height=2.5 *inch)
    im.hAlign = 'CENTER'
    return im
def ActEyeImage(file_path):
    im = Image(file_path, width=4 *inch, height=5.8*inch)
    im.hAlign = 'RIGHT'
    return im
def CutEyeImage(file_path):
    im = Image(file_path, width=4 *inch, height=6.6*inch)
    im.hAlign = 'RIGHT'
    return im
def Gaze9EyeImage(file_path):
    im = Image(file_path, width=7.2 *inch, height=2.5*inch)
    im.hAlign = 'CENTER'
    return im
def Gaze9EyeMesh(file_path):
    im = Image(file_path, width=6/1.2 *inch, height=3/1.2 *inch)
    im.hAlign = 'CENTER'
    return im
def QRCodeImage(file_path):
    im = Image(file_path, width=2 *inch, height=2*inch)
    im.hAlign = 'LEFT'
    return im
def foot1(canvas, can):
    page = "Page {}".format(can.page)
    canvas.saveState()
    canvas.setFont('TSans', 10)
    canvas.setFillColorRGB(.5,.5,.5)
    canvas.drawString((can.width+len(page)*7) / 2, 0.5 * inch, page)
    canvas.restoreState()
    note = "Copyright © 2022 Neurobit Technologies Co., Ltd. All rights reserved."
    canvas.saveState()
    canvas.setFillColorRGB(.5,.5,.5)
    canvas.setFont("Helvetica",6) #choose your font type and font size
    canvas.drawString((can.width*0.75), (can.height*0.05), note)
    logo_path = os.path.join(os.getcwd().replace("\\Result",""), 'logo.png')
    logo = pImage.open(logo_path)
    logo_width, logo_height = logo.size
    n_height = logo_height * 80 / logo_width
    canvas.drawImage(logo_path, 0, int(A4[1]-n_height), width=80, height=n_height)
    canvas.restoreState()
    logo.close()
def CreatePDF(file_path):
    can = BaseDocTemplate(file_path, 
                          pagesize=A4,
                          rightMargin=nb.H_MARGIN,leftMargin=nb.H_MARGIN,
                          topMargin=nb.T_MARGIN, bottomMargin=nb.B_MARGIN)     
    frameT = Frame(can.leftMargin, can.bottomMargin, can.width, can.height, id='normal')
    can.addPageTemplates([PageTemplate(id='OneCol', frames=frameT, onPage=foot1),])
    
    
    return can
def ACTReport(Element, ACT_Task):
    sub_head2 = sub_head("ACT Dynamic Eyeposition Tracking")
    sub_head3 = sub_head("Ocular Alignment -- Alternated Cover Test Sequence in Primary Position")
    #text1 = con_text("Alternated Cover Test Sequence in Primary Position")
    Quality_Bar = quality_bar(ACT_Task.OD, ACT_Task.OS, ACT_Task)
    gaze_table = diagnose_table(ACT_Task)
    
    Element.append(sub_head2)
    im1 = EyeTrackImage(ACT_Task.saveImage_path+"\\DrawEyeTrack.png")
    Element.append(im1)
    Element.append(Quality_Bar)
    Element.append(sub_head3)
    #Element.append(text1)    
    im2 = ActEyeImage(ACT_Task.saveImage_path+"\\DrawEyeFig.png")
    #im3 = QRCodeImage(ACT_Task.saveImage_path+"\\QR_code.png")
# =============================================================================
#     tbl_data = [
#         [gaze_table, im2],
#         [im3,        " "],
#     ]
# =============================================================================
    tbl_data = [
        [im2],
    ]
    style = [
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),        # 對齊
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),        # 對齊
        ('VALIGN', (0, 0), (1, -1), 'CENTER'),        # 對齊
        ('SPAN',(1,0),(1,-1))
        ]
    tbl = Table(tbl_data,colWidths = nb.TABLE_WIDTH *inch/2, style=style)
    Element.append(tbl)
    return Element

def CUTReport(Element, CUT_Task):
    sub_head2 = sub_head("CUT Dynamic Eyeposition Tracking")
    sub_head3 = sub_head("Ocular Alignment -- Cover Uncover Test Sequence in Primary Position")
    #text1 = con_text("Cover Uncover Test Sequence in Primary Position")
    Quality_Bar = quality_bar(CUT_Task.OD, CUT_Task.OS, CUT_Task)
    gaze_table = diagnose_table(CUT_Task)
    
    Element.append(sub_head2)
    im1 = EyeTrackImage(CUT_Task.saveImage_path+"\\DrawEyeTrack.png")
    Element.append(im1)
    Element.append(Quality_Bar)
    Element.append(sub_head3)
    #Element.append(text1)    
    im2 = CutEyeImage(CUT_Task.saveImage_path+"\\DrawEyeFig.png")
    im3 = QRCodeImage(CUT_Task.saveImage_path+"\\QR_code.png")
# =============================================================================
#     tbl_data = [
#         [gaze_table, im2],
#         [im3,        " "],
#     ]
# =============================================================================
    tbl_data = [
        [im2],
    ]
    
    style = [
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),        # 對齊
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),        # 對齊
        ('VALIGN', (0, 0), (1, -1), 'CENTER'),        # 對齊
        ('SPAN',(1,0),(1,-1))
        ]
    tbl = Table(tbl_data,colWidths = nb.TABLE_WIDTH *inch/2, style=style)
    Element.append(tbl)
    return Element

def Gaze9Report(Element, Gaze9_Session):
    GAZE_9_STR      = ["D",       "F",    "L",     
                   "LD",  "LU",  "R",  
                   "RD", "RU", "U"]
    sub_head2 = sub_head("9 Gaze Dynamic Eyeposition Tracking")
    sub_head3 = sub_head("Ocular Motility -- 9 Gaze Test Sequence")
    #text1 = con_text("9 Gaze Test Sequence")
    Quality_Bar = quality_bar(Gaze9_Session.OD, Gaze9_Session.OS, Gaze9_Session)
    
    im1 = EyeTrackImage(Gaze9_Session.saveImage_path+"\\DrawEyeTrack.png")
    im2 = Gaze9EyeImage(Gaze9_Session.saveImage_path+"\\DrawEyeFig.png")
    im3 = Gaze9EyeMesh(Gaze9_Session.saveImage_path+"\\DrawEyeMesh.png")
    #im4 = QRCodeImage(Gaze9_Session.saveImage_path+"\\QR_code.png")
    Dev_H = Gaze9_Session.NeurobitDxDev_H
    Dev_V = Gaze9_Session.NeurobitDxDev_V
    Diff_H = Dev_H[:,0]-Dev_H[:,1]
    Diff_V = Dev_V[:,0]-Dev_V[:,1]
    Gaze9_Session.Diff_H = Diff_H
    Gaze9_Session.Diff_V = Diff_V
    data = [
        ["9 Gaze",  "OD ("+chr(176)+")",   "",        "OS ("+chr(176)+")",  "",     "OD-OS ("+chr(176)+")",    ""],
        ["",        "H",        "V",        "H",        "V",    "H",            "V"],
        [GAZE_9_STR[0], Dev_H[0][0], Dev_V[0][0], Dev_H[0][1], Dev_V[0][1], np.round(Diff_H[0],1), np.round(Diff_V[0],1)],
        [GAZE_9_STR[1], Dev_H[1][0], Dev_V[1][0], Dev_H[1][1], Dev_V[1][1], np.round(Diff_H[1],1), np.round(Diff_V[1],1)],
        [GAZE_9_STR[2], Dev_H[2][0], Dev_V[2][0], Dev_H[2][1], Dev_V[2][1], np.round(Diff_H[2],1), np.round(Diff_V[2],1)],
        [GAZE_9_STR[3], Dev_H[3][0], Dev_V[3][0], Dev_H[3][1], Dev_V[3][1], np.round(Diff_H[3],1), np.round(Diff_V[3],1)],
        [GAZE_9_STR[4], Dev_H[4][0], Dev_V[4][0], Dev_H[4][1], Dev_V[4][1], np.round(Diff_H[4],1), np.round(Diff_V[4],1)],
        [GAZE_9_STR[5], Dev_H[5][0], Dev_V[5][0], Dev_H[5][1], Dev_V[5][1], np.round(Diff_H[5],1), np.round(Diff_V[5],1)],
        [GAZE_9_STR[6], Dev_H[6][0], Dev_V[6][0], Dev_H[6][1], Dev_V[6][1], np.round(Diff_H[6],1), np.round(Diff_V[6],1)],
        [GAZE_9_STR[7], Dev_H[7][0], Dev_V[7][0], Dev_H[7][1], Dev_V[7][1], np.round(Diff_H[7],1), np.round(Diff_V[7],1)],
        [GAZE_9_STR[8], Dev_H[8][0], Dev_V[8][0], Dev_H[8][1], Dev_V[8][1], np.round(Diff_H[8],1), np.round(Diff_V[8],1)],
    ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),    # 字體
        ('FONTSIZE', (0, 0), (-1, -1), 8),          # 字體大小
        ('TEXTCOLOR', (0, 0),(-1, -1), colors.black),          # 字體顏色
        
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('SPAN',(0,0),(0,1)),
        ('SPAN',(1,0),(2,0)),
        ('SPAN',(3,0),(4,0)),
        ('SPAN',(5,0),(6,0)),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),   # 設置表格框線爲grey色，線寬爲0.5
        
        ('BACKGROUND', (0, 0), (-1, -1), colors.white)
    ]    
    gaze_table = Table(dis_list, style=style, rowHeights=12, colWidths=6/8.4 *inch) 
# =============================================================================
#     tbl_data = [
#         [im4, gaze_table]]
#     tbl = Table(tbl_data)
# =============================================================================
    
    Element.append(sub_head2)    
    Element.append(im1)
    Element.append(Quality_Bar)
    Element.append(sub_head3)
    #Element.append(text1)    
    Element.append(im2) 
    Element.append(im3)
    Element.append(gaze_table)
    #Element.append(tbl)
    return Element     
    
def VFReport(Element, VF_Task):
    sub_head2 = sub_head("Pupil Eye Tracking")
    im1 = Image(VF_Task.saveImage_path+"\\DrawPupil.png", width=nb.TABLE_WIDTH * inch *0.9, height=nb.TABLE_WIDTH *1/4 * inch*0.95)
    im_figure = Image(VF_Task.saveImage_path+"\\DrawEyeTrack.png", width=nb.TABLE_WIDTH * inch *0.9, height=nb.TABLE_WIDTH *1/2 * inch*0.95) 
    im_gridscale = Image(os.path.join(VF_Task.major_path, 'gridscale.png'), width=nb.TABLE_WIDTH * inch, height=nb.TABLE_WIDTH*88/1280*inch)
    """Draw pupil table"""
    data = [[         'Pupil Size' , 'OD', 'OS', 'Diff', 'MAE'],
            ['Mean (mm)' , VF_Task.result['Mean']['Right'], VF_Task.result['Mean']['Left'], VF_Task.result['Mean']['Diff_label'], VF_Task.result['Mean']['Diff']],
            ['Min (mm)'  , VF_Task.result['Min']['Right'],  VF_Task.result['Min']['Left'],  VF_Task.result['Min']['Diff_label'], VF_Task.result['Min']['Diff']],
            ['Max (mm)'  , VF_Task.result['Max']['Right'],  VF_Task.result['Max']['Left'],  VF_Task.result['Max']['Diff_label'], VF_Task.result['Max']['Diff']],
            ['Std (mm)'  , VF_Task.result['Std']['Right'],  VF_Task.result['Std']['Left'],  VF_Task.result['Std']['Diff_label'], VF_Task.result['Std']['Diff']],
    ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [('GRID', (0, 0), (-1, -1), .6, colors.black),
             ('TEXTCOLOR', (1, 0), (1, -1), nb.line_color_palatte['reds'][2]),
             ('TEXTCOLOR', (2, 0), (2, -1), nb.line_color_palatte['blues'][2]),
             ('FONTSIZE', (0, 0), (-1, -1), 10), # 字體大小
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ]
    colWidths = (nb.TABLE_WIDTH/5) * inch *0.8   # 每列的寬度
    rowHeights = (nb.TABLE_HEIGHT / 40) * inch    
    component_table1 = Table(dis_list,  colWidths = colWidths, rowHeights=rowHeights, 
            style=style)
    
    tbl_list = [[component_table1], [im1]]
    style = [('ALIGN', (0, 0), (-1, -1), 'CENTER'),
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ]
    colWidths = (nb.TABLE_WIDTH) * inch   # 每列的寬度
    component_table2 = Table(tbl_list, ##colWidths = colWidths, rowHeights=rowHeights*5, 
        style=style)
    
    Element.append(component_table2)
    Element.append(sub_head2)
    Element.append(im_gridscale)
    Element.append(Spacer(1, inch * 0.10))
    Element.append(im_figure)

    return Element
