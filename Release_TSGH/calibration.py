# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:34:54 2022

@author: luc40
"""

import tkinter as tk
import matplotlib.pyploy as plt
from matplotlib.widget import Cursor
from PIL import Image, ImageTk

window = tk.Tk()
window.title('window')
window.geometry('1280x600')
def create_label_image():
    img = Image.open('D:\\Neurobit_Project\\Release\\Result\\C123456789\\20220213_C123456789\\test.jpg')                    
    # 讀取圖片

# =============================================================================
#     img = img.resize( (img.width // 10, img.height // 10) )   
#     # 縮小圖片
# =============================================================================

    imgTk =  ImageTk.PhotoImage(img)                        
    # 轉換成Tkinter可以用的圖片

    lbl_2 = tk.Label(window, image=imgTk)                   
    # 宣告標籤並且設定圖片

    lbl_2.image = imgTk
    lbl_2.grid(column=0, row=0)                             
    # 排版位置

create_label_image()
window.mainloop()
