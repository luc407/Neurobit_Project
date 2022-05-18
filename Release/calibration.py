# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:34:54 2022

@author: luc40
"""

import os
import cv2
import time
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""Initial Setting"""
my_dpi = 600
my_image = 'D:\\Neurobit_Project\\Release\\test.jpg'

home: int = 0
visitor: int = 0
geo_size = '1320x640'

pic = 60
csv_path="D:\\Neurobit_Project\\Release\\Result\\A12345678\\20220308_A12345678\\20220308_162040_A12345678_OcularMotility.csv"

def GetVideo(csv_path):
    fall = csv_path.replace(".csv",".avi")
    if not os.path.isfile(fall):
        fall = csv_path.replace(".csv",".mp4") 
    return cv2.VideoCapture(fall)

cap = GetVideo(csv_path)
cap.set(1,pic)
ret, im = cap.read()
            
class CalibSystem:
    def __init__(self, master):
        self.master = master
        self.xy = []
        master.title("Calibration")
        master.geometry(geo_size)
        
        self.fig = plt.figure(figsize=(1280/my_dpi, 480/my_dpi), dpi=my_dpi,frameon=False)
        self.ax1 = self.fig.add_axes([0, 0, 1, 1])
        self.ax1.imshow(im)
        self.ax1.axis('off')
        self.ax1.text(20,460, "Right",fontsize=5)
        self.ax1.text(1180,460, "Left",fontsize=5)
        self.im1 = FigureCanvasTkAgg(self.fig, master)
        self.im1.get_tk_widget().place(x=20, y=50)        
        self.scat = self.ax1.scatter([],[], s=10, c="b", marker='+',linewidth=.3)
        """Define Cursor"""
        self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        
        """Create Annotation Box"""
        self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
        self.annot.set_visible(False)
        self.fig.canvas.draw()
        
               
        
        self.startCalibrate_button = tk.Button(master, text='Start Calibrate!', command=self.startCalibrate)#.place(relx=0.45, rely=0.9)
        self.startCalibrate_button.pack(side=tk.LEFT, padx=258, pady=50, anchor=tk.S)
        
        self.pre10Frame_button = tk.Button(master, text="<<", command=self.pre10Frame)#.place(relx=0.35, rely=0.9)
        self.pre10Frame_button.pack(side = tk.LEFT, anchor=tk.S, pady=50)
        
        self.preFrame_button = tk.Button(master, text="<", command=self.preFrame)#.place(relx=0.4, rely=0.9)
        self.preFrame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)        
        
        self.nextFrame_button = tk.Button(master, text=">", command=self.nextFrame)#.place(relx=0.55, rely=0.9)
        self.nextFrame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
        
        self.next10Frame_button = tk.Button(master, text=">>", command=self.next10Frame)#.place(relx=0.6, rely=0.9)
        self.next10Frame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)       
    
    def __call__(self, event):
        if event.inaxes is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            self.xy.append([x,y])
            self.annot.xy = (x,y)
            text = "({:d},{:d})".format(x,y)
            self.annot.set_text(text)
            self.annot.set_visible(True)
            try: self.scat.remove() 
            except: pass
            self.scat = self.ax1.scatter(np.array(self.xy)[:,0],np.array(self.xy)[:,1], s=10, c="b", marker='+',linewidth=.3)
            self.fig.canvas.draw()
            
            if len(self.xy)==12:
                self.done_button = tk.Button(self.master, text="Done!", command=self.done)
                self.done_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
            else:
                try: self.done_button.destroy()
                except: pass
                
            if len(self.xy) == 0: self.textvar.set_text("Select right margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 1: self.textvar.set_text("Select left margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 2: self.textvar.set_text("Select peak of upper eyelid of OD."); self.fig.canvas.draw()
            elif len(self.xy) == 3: self.textvar.set_text("Select valley of lower eyelid of OD."); self.fig.canvas.draw()
            elif len(self.xy) == 4: self.textvar.set_text("Select outer corner of OD."); self.fig.canvas.draw()
            elif len(self.xy) == 5: self.textvar.set_text("Select inner corner of OD."); self.fig.canvas.draw()            
            elif len(self.xy) == 6: self.textvar.set_text("Select right margin of OS iris."); self.fig.canvas.draw()
            elif len(self.xy) == 7: self.textvar.set_text("Select left margin of OS iris."); self.fig.canvas.draw()            
            elif len(self.xy) == 8: self.textvar.set_text("Select peak of upper eyelid of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 9: self.textvar.set_text("Select valley of lower eyelid of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 10: self.textvar.set_text("Select inner corner of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 11: self.textvar.set_text("Select outer corner of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 12: self.textvar.set_text("Well done!."); self.fig.canvas.draw()
            elif len(self.xy) > 12: self.textvar.set_text("Too many point!."); self.fig.canvas.draw()
        else:
            print ('Clicked ouside axes bounds but inside plot window')
            
    def popBackXY(self):  
        if len(self.xy)>0:
            self.xy.pop()   
            self.scat.remove()
            if len(self.xy)>0: self.scat = self.ax1.scatter(np.array(self.xy)[:,0],np.array(self.xy)[:,1], s=10, c="b", marker='+',linewidth=.3)
            self.fig.canvas.draw()
            
            if len(self.xy)==12:
                self.done_button = tk.Button(self.master, text="Done!", command=self.done)
                self.done_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
            else:
                try: self.done_button.destroy()
                except: pass
            
            if len(self.xy) == 0: self.textvar.set_text("Select right margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 1: self.textvar.set_text("Select left margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 2: self.textvar.set_text("Select peak of upper eyelid of OD."); self.fig.canvas.draw()
            elif len(self.xy) == 3: self.textvar.set_text("Select valley of lower eyelid of OD."); self.fig.canvas.draw()
            elif len(self.xy) == 4: self.textvar.set_text("Select outer corner of OD."); self.fig.canvas.draw()
            elif len(self.xy) == 5: self.textvar.set_text("Select inner corner of OD."); self.fig.canvas.draw()            
            elif len(self.xy) == 6: self.textvar.set_text("Select right margin of OS iris."); self.fig.canvas.draw()
            elif len(self.xy) == 7: self.textvar.set_text("Select left margin of OS iris."); self.fig.canvas.draw()            
            elif len(self.xy) == 8: self.textvar.set_text("Select peak of upper eyelid of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 9: self.textvar.set_text("Select valley of lower eyelid of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 10: self.textvar.set_text("Select inner corner of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 11: self.textvar.set_text("Select outer corner of OS."); self.fig.canvas.draw()
            elif len(self.xy) == 12: self.textvar.set_text("Well done!."); self.fig.canvas.draw()
            elif len(self.xy) > 12: self.textvar.set_text("Too many point!."); self.fig.canvas.draw()
    
    def done(self):
        self.master.destroy()
    
    def nextFrame(self):
        global pic
        pic+=1
        print(pic)
        if pic < int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            cap.set(1,pic)
            ret, im = cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "Right",fontsize=5)
            self.ax1.text(1180,460, "Left",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
        else:
            pic-=1
    
    def next10Frame(self):
        global pic
        pic+=10
        print(pic)
        if pic < int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            cap.set(1,pic)
            ret, im = cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "Right",fontsize=5)
            self.ax1.text(1180,460, "Left",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
        else:
            pic-=10
            
    def preFrame(self):
        global pic
        pic-=1
        print(pic)
        if pic >= 0 :
            cap.set(1,pic)
            ret, im = cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "Right",fontsize=5)
            self.ax1.text(1180,460, "Left",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
        else:
            pic+=1
    
    def pre10Frame(self):
        global pic
        pic-=10
        print(pic)
        if pic >= 0 :
            cap.set(1,pic)
            ret, im = cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "Right",fontsize=5)
            self.ax1.text(1180,460, "Left",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
        else:
            pic+=10
        
    
        
    def startCalibrate(self):
        self.startCalibrate_button.destroy()
        self.pre10Frame_button.destroy()
        self.preFrame_button.destroy()
        self.nextFrame_button.destroy()
        self.next10Frame_button.destroy()
        
        self.textvar = self.ax1.text(640,40, 
                                    "Select right margin of OD iris.",
                                    fontsize=4, va='center', ha='center')
        self.fig.canvas.draw()
        
        self.popBack_button = tk.Button(self.master, text="Back", command=self.popBackXY)
        self.popBack_button.pack(side=tk.LEFT, padx=400, anchor=tk.S, pady=50) 
        
        
        self.fig.canvas.callbacks.connect('button_press_event', self)
        

        
        
if __name__ == "__main__":
    root = tk.Tk()
    my_gui = CalibSystem(root)
    root.mainloop()
    