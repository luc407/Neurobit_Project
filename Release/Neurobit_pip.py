# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 20:05:52 2022

@author: luc40
"""
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('qrcode')
install('PyDrive')
install('moviepy')
install('google-cloud-storage')
install('reportlab')
install('glob2')
install('pytest-shutil')
install('opencv-python')
install('Pillow')
install('cmake')
install('boost')
install('tqdm')
install('matplotlib')
install('pandas')
install('scipy')
install('ttkbootstrap')

import os
import glob
import matplotlib
import shutil
from shutil import copyfile
import matplotlib.font_manager
folder = matplotlib.__file__
folder = folder.replace("__init__.py","mpl-data\\fonts\\ttf")
font1_src = os.getcwd()+"\\TaipeiSansTCBeta-Regular.ttf"
font2_src = os.getcwd()+"\\TaipeiSansTCBeta-Bold.ttf"
font3_src = os.getcwd()+"\\arial.ttf"
font4_src = os.getcwd()+"\\arialbd.ttf"
font1_dst = os.path.join(folder,"TaipeiSansTCBeta-Regular.ttf")
font2_dst = os.path.join(folder,"TaipeiSansTCBeta-Bold.ttf")
font3_dst = os.path.join(folder,"arial.ttf")
font4_dst = os.path.join(folder,"arialbd.ttf")
copyfile(font1_src, font1_dst)
copyfile(font2_src, font2_dst)
copyfile(font3_src, font3_dst)
copyfile(font4_src, font4_dst)

json_path = glob.glob("C:\\Users\\*\\.matplotlib\\*.json")
for json in json_path:
    print(json)
    os.remove(json)
import matplotlib
os._exit(00)

