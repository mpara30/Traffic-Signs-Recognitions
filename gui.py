import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model

model = load_model('traffic_classifier.h5')

classes = {1: 'Limita de viteza (20km/h)',
           2: 'Limita de viteza (30km/h)',
           3: 'Limita de viteza (50km/h)',
           4: 'Limita de viteza (60km/h)',
           5: 'Limita de viteza (70km/h)',
           6: 'Limita de viteza (80km/h)',
           7: 'Sfarsitul limitei de viteza (80km/h)',
           8: 'Limita de viteza (100km/h)',
           9: 'Limita de viteza (120km/h)',
           10: 'Nu treceti',
           11: 'Nu trec vehicule cu greutate mai mare de 3.5 tone',
           12: 'Drept de trecere la intersectie',
           13: 'Drum cu prioritate',
           14: 'Acordati prioritate',
           15: 'Stop',
           16: 'Fara vehicule',
           17: 'Vehicule cu greutate mai mare de 3.5 tone interzise',
           18: 'Nu se intra',
           19: 'Atentie generala',
           20: 'Curba periculoasa la stanga',
           21: 'Curba periculoasa la dreapta',
           22: 'Curba dubla',
           23: 'Drum accidentat',
           24: 'Drum alunecos',
           25: 'Drumul se ingusteaza pe dreapta',
           26: 'Drum in lucru',
           27: 'Semafor',
           28: 'Pietoni',
           29: 'Trec copii',
           30: 'Trec biciclete',
           31: 'Atentie la gheata/zapada',
           32: 'Trec animale salbatice',
           33: 'End speed + passing limits',
           34: 'Virati la dreapta inainte',
           35: 'Virati la stanga inainte',
           36: 'Inainte',
           37: 'Mergeti drept sau la dreapta',
           38: 'Mergeti drept sau la stanga',
           39: 'Tineti dreapta',
           40: 'Tineti stanga',
           41: 'Sens giratoriu',
           42: 'End of no passing',
           43: 'End no passing vehicle with a weight greater than 3.5 tons'}

# GUI Initialisation

top = tk.Tk()
top.geometry('800x600')
top.title('Clasificarea semnelor de circulatie')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict([image])[0]
    sign = classes[numpy.argmax(pred)+1]
    print(sign)
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text='Clasificati imaginea', command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Incarcati o imagine", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text='Verificati semnul de circulatie', pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')

heading.pack()
top.mainloop()
