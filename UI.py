from Bonus import mainBackPropagation
import tkinter as tKinter
import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sn

window = tKinter.Tk()
window.geometry("700x200")

# num layers label
num_layers_label = tKinter.Label(window, text="Layers num", font=("Times", 10))
num_layers_label.place(x=25, y=35)
num_layers_label.configure(bg='white')
# num_Layers Text Box
num_layers_text = tKinter.Entry(window, width=10)
num_layers_text.pack()
num_layers_text.configure(bg='white')
num_layers_text.place(x=110, y=38)

# neuron num label
neurons_num_label = tKinter.Label(window, text="Neurons num", font=("Times", 10))
neurons_num_label.place(x=220, y=35)
neurons_num_label.configure(bg='white')
# num_Layers Text Box
num_neurons_text = tKinter.Entry(window, width=10)
num_neurons_text.pack()
num_neurons_text.configure(bg='white')
num_neurons_text.place(x=320, y=38)

# learning rate label
learning_rate_label = tKinter.Label(window, text="Learning rate", font=("Times", 10))
learning_rate_label.place(x=450, y=35)
learning_rate_label.configure(bg='white')
# num_Layers Text Box
learning_rate_text = tKinter.Entry(window, width=10)
learning_rate_text.configure(bg='white')
learning_rate_text.pack()
learning_rate_text.place(x=550, y=38)

# num layers label
num_epochs_label = tKinter.Label(window, text="Epochs num", font=("Times", 10))
num_epochs_label.place(x=25, y=105)
num_epochs_label.configure(bg='white')
# num_Layers Text Box
num_epochs_text = tKinter.Entry(window, width=10)
num_epochs_text.configure(bg='white')
num_epochs_text.pack()
num_epochs_text.place(x=110, y=108)

# Bias check box
bias_var=tKinter.IntVar()
bias_check = tKinter.Checkbutton(window, text="Bias", font=("times", 10),variable=bias_var)
bias_check.configure(bg='white')
bias_check.place(x=320, y=108)

# Acivation Function options
optionsVar = tKinter.StringVar(window)
optionsVar.set("Sigmoid")
function_menu = tKinter.OptionMenu(window, optionsVar, "Sigmoid", "Tangent Sigmoid")
function_menu.configure(bg='white')
function_menu.pack()
function_menu.place(x=500, y=108)

def draw_confusion_mat(confusion_matrix):
    fig = plt.gcf()
    fig.canvas.set_window_title('Confusion Matrix')
    plt.clf()
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

def mainFun():
    if not num_neurons_text.get() or not num_layers_text.get() or not num_epochs_text.get() or not learning_rate_text.get() :
        messagebox.showerror("Warning !!!", "can't accept empty strings ")
    else :
        num_neurons = list(map(int,(num_neurons_text.get()).split(',')))
        num_layers = int(num_layers_text.get())
        num_epochs = int(num_epochs_text.get())
        num_learing_rate = float(learning_rate_text.get())
        application_function=optionsVar.get()
        #num_neurons,num_layers,num_epochs,num_learing_rate,bias,application_function
        confusion_matrix,accuracy= mainBackPropagation(num_neurons,num_layers,num_epochs,num_learing_rate,
                                                       int(int(bias_var.get())*np.random.randn(1)),application_function)
        messagebox.showinfo("Accuracy", "Accuracy is : "+str(accuracy))
        draw_confusion_mat(confusion_matrix)


    # Run Training Button
run_train_Btn = tKinter.Button(window, text="Train", width=10, bg='cyan', font=("times", 10), command=mainFun)
run_train_Btn.place(x=275, y=160)



window.configure(bg='white')
window.title("Digits Recognition")
window.mainloop()
