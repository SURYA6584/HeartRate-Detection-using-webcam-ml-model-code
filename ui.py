import os
from tkinter import *
from PIL import Image,ImageTk

win=Tk()
win.title("Heart rate Detector ")
win.geometry("500x500")
win.configure(bg="cyan")
win.resizable(False,False)

img=Image.open("bag.webp")
img=img.resize((500,500))
bgg=ImageTk.PhotoImage(img)

label=Label(win,image=bgg)
label.place(x=0,y=0)

title=Label(win,text="Heart Rate Finder  From Face",font=("times",20,"bold italic"))
title.place(x=70,y=10)

def start():
    os.system("python real_time_hb.py")
btn=Button(win,text="Start Calculate",font=("times",20,"bold italic"),command=start,activebackground="green")
btn.place(x=180,y=350)
