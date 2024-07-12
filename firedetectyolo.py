import tkinter as tk
import customtkinter as ctk # tkinter extension which provides extra ui-elements

import torch
import numpy as np
from playsound import playsound
from pathlib import Path #path file sound
import os

import cv2
from PIL import Image, ImageTk # PIL untuk memanipulasi file gambar
# import vlc # media player
# import threading # mengatsi not responding pada window


app = tk.Tk()
app.geometry("416x416")
app.title("Fire Detector PPSDM Migas Cepu")
ctk.set_appearance_mode("dark") # darkmode

vidFrame = tk.Frame(height=416, width=416)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

model = torch.hub.load('ultralytics/yolov5','custom', path='C:\\Users\\asus\\Flask_Python\\myenv\\Lib\\site-packages\\yolov5\\best.pt', force_reload=True)

# model = load_state_dict(torch.load('Lib/site-packages/yolov5/best.pt'))

cap = cv2.VideoCapture(0)
def detect():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())

    print(results.xywh[0])

    if len(results.xywh[0]) > 0:
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]

        if dconf.item() > 0.70 and dclass.item() == 0.0 :
            # path = 'C:\\Users\\asus\\Flask_Python\\myenv\\sound'
            # os.chdir(path)
            # sound = Path().cwd() / "beep.mp3"
            # playsound(sound)
            playsound('beep.mp3')
            # sound = vlc.MediaPlayer("beep.mp3")
            # sound.play()

    imgarr = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    vid.after(10, detect)

detect()
app.mainloop()







# th = threading.Thread(target=detect)
# th

# on_button = tk.Button(app, 
#                     text="Run",
#                     command=threading.Thread(target=detect).start)




