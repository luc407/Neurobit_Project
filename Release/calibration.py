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
home: int = 0
visitor: int = 0
geo_size = '1320x640'
pic = 60
Fs = 30

def GetVideo(csv_path):
    fall = csv_path.replace(".csv",".avi")
    if not os.path.isfile(fall):
        fall = csv_path.replace(".csv",".mp4") 
    return cv2.VideoCapture(fall)
            
class CalibSystem:
    def __init__(self, csv_path):
        self.master = tk.Tk()
        self.xy = []
        self.master.title("Calibration")
        self.master.geometry(geo_size)
        
        self.csv_path = csv_path
        self.cap = GetVideo(self.csv_path)
        self.cap.set(1,pic)
        ret, im = self.cap.read()
        self.width = im.shape[1]
        self.height = im.shape[0]

        self.fig = plt.figure(figsize=(self.width/my_dpi, self.height/my_dpi), dpi=my_dpi,frameon=False)
        self.ax1 = self.fig.add_axes([0, -0.15, 1, 1.2])
        self.ax1.imshow(im)
        self.ax1.set_xlim(0,self.width)
        self.ax1.set_ylim(self.height-40,40)        
        self.ax1.axis('off')
        self.ax1.text(int(self.width/4),self.height-60, "OD",fontsize=5)
        self.ax1.text(int(self.width*3/4),self.height-60, "OS",fontsize=5)
        self.im1 = FigureCanvasTkAgg(self.fig, self.master)
        self.im1.get_tk_widget().place(x=20, y=50)        
        self.scat = self.ax1.scatter([],[], s=10, c="b", marker='+',linewidth=.3)
        """Define Cursor"""
        self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        
        """Create Annotation Box"""
        self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
        self.annot.set_visible(False)
        self.fig.canvas.draw()
        
        self.startCalibrate_button = tk.Button(self.master, text='Start Calibration!', font=('Arial',15,'bold'), command=self.startCalibrate)#.place(relx=0.45, rely=0.9)
        self.startCalibrate_button.pack(side=tk.LEFT, padx=258, pady=50, anchor=tk.S)
        
        self.pre10Frame_button = tk.Button(self.master, text="<<", font=('Arial',15,'bold'), command=self.pre10Frame)#.place(relx=0.35, rely=0.9)
        self.pre10Frame_button.pack(side = tk.LEFT, anchor=tk.S, pady=50)
        
        self.preFrame_button = tk.Button(self.master, text="<", font=('Arial',15,'bold'), command=self.preFrame)#.place(relx=0.4, rely=0.9)
        self.preFrame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)        
        
        self.nextFrame_button = tk.Button(self.master, text=">", font=('Arial',15,'bold'), command=self.nextFrame)#.place(relx=0.55, rely=0.9)
        self.nextFrame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
        
        self.next10Frame_button = tk.Button(self.master, text=">>", font=('Arial',15,'bold'), command=self.next10Frame)#.place(relx=0.6, rely=0.9)
        self.next10Frame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)   
        
        self.text = tk.Label(self.master, text="Timestamp: "+str(round(pic/Fs,2))+"s", font=('Arial',15,'bold'))
        self.text.pack(side=tk.RIGHT, anchor=tk.S, pady=60, padx=50)

        self.master.mainloop()
    
    def __call__(self, event):
        if event.inaxes is not None:
            x = int(event.xdata)            
            
            if len(self.xy) == 0:   y = int(event.ydata); self.cursor = Cursor(self.ax1, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=.3)
            elif len(self.xy) == 1: y = self.xy[0][1]
            elif len(self.xy) == 2: y = self.xy[0][1]; self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            elif len(self.xy) == 3: y = int(event.ydata); self.cursor = Cursor(self.ax1, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=.3)
            elif len(self.xy) == 4: y = self.xy[3][1]
            elif len(self.xy) == 5: y = self.xy[3][1]; self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            else: return
           
            self.xy.append([x,y])
            self.annot.xy = (x,y)
            text = "({:d},{:d})".format(x,y)
            self.annot.set_text(text)
            self.annot.set_visible(True)
            try: self.scat.remove() 
            except: pass
            self.scat = self.ax1.scatter(np.array(self.xy)[:,0],np.array(self.xy)[:,1], s=10, c="b", marker='+',linewidth=.3)
            self.fig.canvas.draw()
            
            if len(self.xy)==6:
                self.done_button = tk.Button(self.master, text="Done!", font=('Arial',15,'bold'), command=self.done)
                self.done_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
            else:
                try: self.done_button.destroy()
                except: pass
                
            if len(self.xy) == 0: self.textvar.set_text("Select the center of OD pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 1: self.textvar.set_text("Select the right margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 2: self.textvar.set_text("Select the left margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 3: self.textvar.set_text("Select the center of OS pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 4: self.textvar.set_text("Select the right margin of OS iris."); self.fig.canvas.draw()
            elif len(self.xy) == 5: self.textvar.set_text("Select the left margin of OS iris."); self.fig.canvas.draw()            
            elif len(self.xy) == 6: self.textvar.set_text("Well done!"); self.fig.canvas.draw()
            elif len(self.xy) > 6: self.textvar.set_text("Too many points on the canvas!"); self.fig.canvas.draw()
        else:
            print ('Clicked ouside axes bounds but inside plot window')
    
    def popBackXY(self):  
        if len(self.xy)>0:
            self.xy.pop()   
            self.scat.remove()
            if len(self.xy)>0: self.scat = self.ax1.scatter(np.array(self.xy)[:,0],np.array(self.xy)[:,1], s=10, c="b", marker='+',linewidth=.3)
            self.fig.canvas.draw()
            
            if len(self.xy)==6:
                self.done_button = tk.Button(self.master, text="Done!", font=('Arial',15,'bold'), command=self.done)
                self.done_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
            else:
                try: self.done_button.destroy()
                except: pass
            
            if len(self.xy) == 0: self.textvar.set_text("Select the center of OD pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 1: self.textvar.set_text("Select the right margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 2: self.textvar.set_text("Select the left margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 3: self.textvar.set_text("Select the center of OS pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 4: self.textvar.set_text("Select the right margin of OS iris."); self.fig.canvas.draw()
            elif len(self.xy) == 5: self.textvar.set_text("Select the left margin of OS iris."); self.fig.canvas.draw()            
            elif len(self.xy) == 6: self.textvar.set_text("Well done!"); self.fig.canvas.draw()
            elif len(self.xy) > 6: self.textvar.set_text("Too many points on the canvas!"); self.fig.canvas.draw()
    
    def done(self, event=None):
        plt.clf() 
        plt.close(self.fig) 
        self.OD_WTW = abs(self.xy[1][0]-self.xy[2][0])
        self.OS_WTW = abs(self.xy[4][0]-self.xy[5][0])
        self.master.quit()
        self.master.destroy()
        for item in self.im1.get_tk_widget().find_all():
           self.im1.get_tk_widget().delete(item)
    
    def nextFrame(self):
        global pic
        pic+=1
        if pic < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,pic)
            ret, im = self.cap.read()
            self.ax1.clear()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(int(self.width/4),self.height-60, "OD",fontsize=5)
            self.ax1.text(int(self.width*3/4),self.height-60, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            
        else:
            pic-=1
        self.text.destroy()
        self.text = tk.Label(self.master, text="Timestamp: "+str(round(pic/Fs,2))+"s", font=('Arial',15,'bold'))
        self.text.pack(side=tk.RIGHT, anchor=tk.S, pady=60, padx=50)
    
    def next10Frame(self):
        global pic
        pic+=10
        if pic < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,pic)
            ret, im = self.cap.read()
            self.ax1.clear()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(int(self.width/4),self.height-60, "OD",fontsize=5)
            self.ax1.text(int(self.width*3/4),self.height-60, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            pic-=10
        self.text.destroy()
        self.text = tk.Label(self.master, text="Timestamp: "+str(round(pic/Fs,2))+"s", font=('Arial',15,'bold'))
        self.text.pack(side=tk.RIGHT, anchor=tk.S, pady=60, padx=50)
            
    def preFrame(self):
        global pic
        pic-=1
        if pic >= 0 :
            self.cap.set(1,pic)
            ret, im = self.cap.read()
            self.ax1.clear()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(int(self.width/4),self.height-60, "OD",fontsize=5)
            self.ax1.text(int(self.width*3/4),self.height-60, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)            
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            pic+=1
        self.text.destroy()
        self.text = tk.Label(self.master, text="Timestamp: "+str(round(pic/Fs,2))+"s", font=('Arial',15,'bold'))
        self.text.pack(side=tk.RIGHT, anchor=tk.S, pady=60, padx=50)
    
    def pre10Frame(self):
        global pic
        pic-=10
        if pic >= 0 :
            self.cap.set(1,pic)
            ret, im = self.cap.read()
            self.ax1.clear()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(int(self.width/4),self.height-60, "OD",fontsize=5)
            self.ax1.text(int(self.width*3/4),self.height-60, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
               
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            pic+=10
        self.text.destroy()
        self.text = tk.Label(self.master, text="Timestamp: "+str(round(pic/Fs,2))+"s", font=('Arial',15,'bold'))
        self.text.pack(side=tk.RIGHT, anchor=tk.S, pady=60, padx=50)
        
    def startCalibrate(self):
        self.startCalibrate_button.destroy()
        self.pre10Frame_button.destroy()
        self.preFrame_button.destroy()
        self.nextFrame_button.destroy()
        self.next10Frame_button.destroy()
        
        self.textvar = self.ax1.text(int(1280/2),0, 
                                    "Select the center of OD pupil.",
                                    fontsize=4, va='center', ha='center', color='#2ea9df')
        self.fig.canvas.draw()
        
        self.popBack_button = tk.Button(self.master, text="Back", font=('Arial',15,'bold'), command=self.popBackXY)
        self.popBack_button.pack(side=tk.LEFT, padx=400, anchor=tk.S, pady=50) 
        
        self.fig.canvas.callbacks.connect('button_press_event', self)

if __name__ == "__main__":
    root = tk.Tk()
    my_gui = CalibSystem(root, csv_path)
    root.mainloop()
    