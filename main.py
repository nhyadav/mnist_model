from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

model = load_model('mnist.h5')

def predict_digit(img):
    # resize the image to 28*28

    imge = img.resize((28, 28))

    # convert rgb to gray scale

    imge = imge.convert('L')
    imge = np.array(imge)

    # reshaping to support our model input and normaling
    imge = imge.reshape(1, 28, 28, 1)
    imge = imge/255.0
    # predicting the class
    # res = model.predict([imge])[0]
    res = model.predict_classes(imge)
    return res[0]



class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # crearting element
        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", cursor="arrow")
        self.label = tk.Label(self, text="Draw..", font="arial 40")
        self.classfy_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.clear_button = tk.Button(self, text="clear", command = self.clear_all)

        # grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classfy_btn.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self) :
        self.canvas.delete("all")

    def classify_handwriting(self):
        hwnd = self.canvas.winfo_id()      # get the handle of canvas
        rect = win32gui.GetWindowRect(hwnd)
        a, b, c, d = rect
        rect = (a+10 , b+10 , c - 4 , d - 4)
        im = ImageGrab.grab(rect)
        im.save("draw5.jpg")
        # digit, acc = predict_digit(im)
        digit = predict_digit(im)
        # self.label.configure(text=str(digit)+", "+str(int(acc*100))+"%")
        self.label.configure(text=str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 6
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill="white")



app = App()
app.mainloop()