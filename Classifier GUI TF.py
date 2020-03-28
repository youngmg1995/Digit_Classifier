# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:32:55 2020

@author: Mitchell

Classifier GUI TF.py
~~~~~~~~~~~~~~~~~~~~
An interactive GUI for classifying user inputted handwritten digits using
neural networks constructed using tensorflow. The GUI allows the user to draw an
image on the canvas and then select one of three simple ANN models to classify 
the image as a digit in 0-9. The user has the option of choosing between three
models described below. The results are then displayed on the GUI interface,
including both the output activations and the final image classification.
Additionally, the user has the option of saving a copy of the classified image
to the file of choice.

Note that this file requires a working python environment with tensorflow.
Specifically, this code was run and tested using tensorflow version 2.0 to
load our models and classify the images. In addition to this, the file also
uses additional libraries, such as tkinter, PIL, etc., to build the interactive
GUI and manipulate images.

The script used to construct and save the below networks is included in the
project under the file name "Network_Builder_TF.py".
    
1) Naive NN:
    Fully connected neural network built with little experimentation and
    trained in minutes thanks to its minimal size. See below for details
    regarding the specific network structure and training parameters of the
    model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Network: 
   -Structure: Single hiddden layer (784,30,10)
   -Activations: Sigmoid neurons for hidden and output layer
   -Initialization: Random starting weights/biases chosen from gaussina distr.
       -Weigths and Biases: mean = 0, std = 1

 Training:
   -Quadratic Cost Function
   -Stochastic Gradient Descent
       -Learning Speed: Eta = 3.0 
   -Trained for 30 epochs with minibatches of size 10

 Performance:
   -Accuracy: ~94.8% (Against MNIST test images)
   -# of Parameters: 23,860
   -Training Time: ~10 minutes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                
2) Optimized NN:
    Another fully connected neural network, but built through more
    experimentation, andavnced techniques, and optimization of parameters in
    order to imporove performance. See below for details regarding the specific
    techniques used, network structure and training parameters of the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Network: 
   -Structure: Two hiddden layers (784,800,800,10)
   -Activations: ReLU hidden layers and SoftMax output layer
   -Initialization: Random starting weights/biases chosen from gaussina distr.
       -Biases: mean = 0, std = 1
       -Weights: mean = 0, std = 1/(# inputs)^(1/2)

 Training:
   -Cross Entropy Cost Function
   -Regularization 
       -max-norm wieght contraint with l=4
       -Dropout: Hidden Layers Rate = .5, Input Layer Rate = .2
   -Momentum Based Stochastic Gradient Descent
       -Friction Coeff = .5
       -Learning Speed: Eta = .5*.995^(t) where t is the epoch number 
   -Trained for 1000 epochs with minibatches of size 100

 Performance:
   -Accuracy: ~98.8%
   -# of Parameters: 1,276,810
   -Training Time: ~10 hours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                
3) Convolution NN:
    Simple convolutional neural network that uses two convolutional layers
    and a single fully connected hidden layer. Convolutional networks have been
    proven to generalize and perform very well on image recognition models.
    See below for details regarding the specific techniques used, network
    structure and training parameters of the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Network: 
   -Structure: Convolutional Neural Network
       -Layer 1: Convolutional Pooling Layer
           -Input: 1 x 28 x 28 image
           -#  of Feature Maps: 20
           -Local Receptive Field: 5 x 5
           -Stride Length: 1
           -Pooling: 2 x 2 input region Max Pooling
       -Layer 2: Convolutional Pooling LAyer
           -Input: 20 x 12 x 12 output of Convo. Layer 1
           -#  of Feature Maps: 20
           -Local Receptive Field: 5 x 5
           -Stride Length: 1
           -Pooling: 2 x 2 input region Max Pooling
       -Layer 3: Dense Neural Network
           -Structure: [640,100,10]
           -Activations: ReLU for 100 hidden nodes, SoftMax for Output Layer
   -Initialization: Random starting weights/biases chosen from gaussina distr.
       -Biases: mean = 0, std = 1
       -Weights: mean = 0, std = 1/(# inputs)^(1/2)

 Training:
   -Cross Entropy Cost Function
   -Regularization 
       -max-norm wieght contraint with l=4
       -Dropout: Hidden Layers Rate = .5, Input Layer Rate = .2
   -Momentum Based Stochastic Gradient Descent
       -Friction Coeff = .5
       -Learning Speed: Eta = .5*.995^(t) where t is the epoch number 
   -Trained for 1000 epochs with minibatches of size 100

 Performance:
   -Accuracy: ~98.9%
   -# of Parameters: 85,670
   -Training Time: ~1 hour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

### Libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import PIL, tkinter, os
from PIL import ImageDraw, ImageTk
from tkinter import ttk, messagebox
import numpy as np


### Loading in Networks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
folder = 'Network_Builder/Saved_Networks/'

filename_1 = 'network1_tf'
net_1 = tf.keras.models.load_model(folder + filename_1)
print('Naive NN loaded')

filename_2 = 'network2_tf'
net_2 = tf.keras.models.load_model(folder + filename_2)
print('Optimized NN loaded')

filename_3 = 'network3_tf'
net_3 = tf.keras.models.load_model(folder + filename_3)
print('Convolution NN loaded')


### Creating Interactive Tkinter GUI
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Defining Window and Dimensions for GUI
#------------------------------------------------------------------------------
header_height = 10
scale = 16
width = 28*scale
height = 28*scale
center = height//2

mainWindow = tkinter.Tk()
mainWindow.title("ANN Digit Classifier")

mainWindow_width = 920
mainWindow_height = 550
header_height = 10
mainWindow.geometry('{}x{}'.format(mainWindow_width, mainWindow_height+header_height))

## Helper Functions for buttons
#------------------------------------------------------------------------------
def save_canvas():
    """
    Saves a resized version of the canvas image to the folder
    "Saved_GUI_Images." Specifically saves the 28 x 28 resized image of the
    convas that is evaluated by the chosen model.
    """
    folder_name = 'Saved_GUI_Images/'
    file_name = savefile_entry.get()
    image2 = image1.copy().resize((28,28),PIL.Image.BILINEAR)
    if file_name == 'Enter Filename Here: ':
        messagebox.showinfo('Save Image Error!!!', 'Please Enter Filename for Image')
    else:
        file_exists = os.path.exists(folder_name+file_name+'.tiff')
        if file_exists:
            replace_file = messagebox.askokcancel('Save Image Error!!!', 'File already exists.\nPress \'OK\' to replace file or \'Cancel\' to cancel save')
            if replace_file:
                image2.save(folder_name+file_name+'.tiff')
        else:
            image2.save(folder_name+file_name+'.tiff')

def clear_canvas():
    """
    Clears the drawing canvas so the user can draw a new digit to classify.
    """
    canvas.delete('all')
    image1.paste(PIL.Image.new("F", (width, height), white))
    
def image_resize(img):
    """
    Function used to format our images for evaluation by our ANN models.
    Just resizes the image to be 28 x 28 using BILENEAR interpolation.
    """
    img2 = img.copy().resize((28,28),PIL.Image.BILINEAR)
    img2 = np.array(img2).reshape((1,28,28))
    return img2

# Images used for displaying model classification output
number_images = []
for i in range(10):
    number_images.append(ImageTk.PhotoImage(PIL.Image.open("Number_Images/180px-{}.png".format(i))))

def classify():
    """
    Function that classifies the canvas image as a digit: 0-9. Does so by
    passing a 28 x 28 resized version of the canvas image into the selected
    ANN model. The model then evauates the image and the reults are displayed
    on the GUI interface. Both the output activations for each digit and the
    final classification are displayed.
    """
    image2 = image_resize(image1)
    selected_classifier = selected.get()
    if selected_classifier == 1:
        result = net_1(image2).numpy().reshape((10,1))
    elif selected_classifier == 2:
        result = net_2(image2).numpy().reshape((10,1))
    elif selected_classifier == 3:
        img = image2.copy().reshape((1,28,28,1))
        result = net_3(img).numpy().reshape((10,1))
    
    bar0.configure(value = result[0][0]*100)
    bar0_value.configure(text = ('%5.4f'%(result[0][0])))
    bar1.configure(value = result[1][0]*100)
    bar1_value.configure(text = ('%5.4f'%(result[1][0])))
    bar2.configure(value = result[2][0]*100)
    bar2_value.configure(text = ('%5.4f'%(result[2][0])))
    bar3.configure(value = result[3][0]*100)
    bar3_value.configure(text = ('%5.4f'%(result[3][0])))
    bar4.configure(value = result[4][0]*100)
    bar4_value.configure(text = ('%5.4f'%(result[4][0])))
    bar5.configure(value = result[5][0]*100)
    bar5_value.configure(text = ('%5.4f'%(result[5][0])))
    bar6.configure(value = result[6][0]*100)
    bar6_value.configure(text = ('%5.4f'%(result[6][0])))
    bar7.configure(value = result[7][0]*100)
    bar7_value.configure(text = ('%5.4f'%(result[7][0])))
    bar8.configure(value = result[8][0]*100)
    bar8_value.configure(text = ('%5.4f'%(result[8][0])))
    bar9.configure(value = result[9][0]*100)
    bar9_value.configure(text = ('%5.4f'%(result[9][0])))
    
    img = number_images[np.argmax(result)]
    image_label.configure(image=img)
    image_label.image = img


def paint(event):
    """
    Function used to draw on the canvas. Does so by creating a circle of radius
    12 pixels at each point the users mouse passes over.
    """
    r = scale*1.5
    x1, y1 = (event.x - r), (event.y - r)
    x2, y2 = (event.x + r), (event.y + r)
    canvas.create_oval(x1, y1, x2, y2, fill="black")
    draw.ellipse([x1, y1, x2, y2],fill="white")
#------------------------------------------------------------------------------

## Building All The Elements of the GUI
#------------------------------------------------------------------------------
# All the elements of the left side of the GUI    
left_frame = tkinter.Frame(mainWindow, width=mainWindow_width/2, height=mainWindow_height)
left_frame.grid(column = 0, row = 0, padx = 10, pady = 10)
left_frame.grid_propagate(0)

# Label above canvas
canvas_label = tkinter.Label(left_frame, text = 'Drawing Canvas', font=('TkDefaultFont', 24, 'bold'))
canvas_label.grid(column = 0, row = 0, columnspan = 3)

# Canvas for drawing numbers 
canvas = tkinter.Canvas(left_frame, relief='raised', borderwidth=1, width=width, height=height, bg='white')
canvas.grid(column = 0, row = 1, columnspan = 3)

# Button for reseting the canvas
clear_canvas_button = tkinter.Button(left_frame, text='Clear Canvas', font=('TkDefaultFont', 12), padx=5, pady=5, command=clear_canvas)
clear_canvas_button.grid(column = 0, row = 2)

# Button for saving canvas image
# Note - Actual image saved is not the canvas exactly. Instead we save the
# resized 28 x 28 canvas image that is classified by each ANN model.
save_canvas_button = tkinter.Button(left_frame, text='Save Image', font=('TkDefaultFont', 12), padx=5, pady=5, command=save_canvas)
save_canvas_button.grid(column = 1, row = 2)

# Entry slot for naming filename for image to be saved
savefile_entry = tkinter.Entry(left_frame, width=20, borderwidth=5, font = ('TkDefaultFont', 12))
savefile_entry.grid(column = 2, row = 2)
savefile_entry.insert(0, 'Enter Filename Here: ')

# All the elements of the right side of the GUI    
right_frame = tkinter.Frame(mainWindow, width=mainWindow_width/2, height=mainWindow_height)
right_frame.grid(column = 1, row = 0, padx = 10, pady = 10)
right_frame.grid_propagate(0)

## Label above ANN models
network_label = tkinter.Label(right_frame, text = 'ANN Classifier Model', font=('TkDefaultFont', 24, 'bold'))
network_label.grid(column = 0, row = 0, columnspan = 10)

# Radio buttons for choosing which ANN classifies the image
selected = tkinter.IntVar(value=1)
radio1 = tkinter.Radiobutton(right_frame, text='Naive NN', font=('TkDefaultFont', 12), value = 1, variable = selected, pady = 10)
radio2 = tkinter.Radiobutton(right_frame, text='Optimized NN', font=('TkDefaultFont', 12), value = 2, variable = selected, pady = 10)
radio3 = tkinter.Radiobutton(right_frame, text='Convolution NN', font=('TkDefaultFont', 12), value = 3, variable = selected, pady = 10)

radio1.grid(column = 0, row = 1, columnspan = 3)
radio2.grid(column = 3, row = 1, columnspan = 4)
radio3.grid(column = 7, row = 1, columnspan = 3)

# Button for evaluating classification of image using chosen model
classify_button = tkinter.Button(right_frame, text='Classify\nImage', bg = 'blue', fg = 'white', font=('TkDefaultFont', 24, 'bold'), padx=10, pady=10, command=classify)
classify_button.grid(column = 0, row = 2, columnspan = 5)

# Classification results
# Specifically an image of the resultant classification/number chosen by or model
image_label = tkinter.Label(right_frame, image=number_images[0], pady = 10)
image_label.grid(column = 5, row = 2, columnspan = 5)

# Label above model activations (array of values for each classification)
activations_label = tkinter.Label(right_frame, text = 'Output Activations', font=('TkDefaultFont', 24, 'bold'), pady = 10)
activations_label.grid(column = 0, row = 3, columnspan = 10)

# Bars and values for each activation
style = ttk.Style()
style.theme_use('default')
style.configure("Vertical.TProgressbar", background='green')
bar0 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar1 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar2 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar3 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar4 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar5 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar6 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar7 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar8 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar9 = ttk.Progressbar(right_frame, orient = 'vertical', length=150, mode = 'determinate', style="Vertical.TProgressbar")
bar0['value'] = 0.
bar1['value'] = 0.
bar2['value'] = 0.
bar3['value'] = 0.
bar4['value'] = 0.
bar5['value'] = 0.
bar6['value'] = 0.
bar7['value'] = 0.
bar8['value'] = 0.
bar9['value'] = 0.

bar0.grid(column = 0, row = 5, padx = 10)
bar1.grid(column = 1, row = 5, padx = 10)
bar2.grid(column = 2, row = 5, padx = 10)
bar3.grid(column = 3, row = 5, padx = 10)
bar4.grid(column = 4, row = 5, padx = 10)
bar5.grid(column = 5, row = 5, padx = 10)
bar6.grid(column = 6, row = 5, padx = 10)
bar7.grid(column = 7, row = 5, padx = 10)
bar8.grid(column = 8, row = 5, padx = 10)
bar9.grid(column = 9, row = 5, padx = 10)

bar0_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar0['value'])))
bar1_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar1['value'])))
bar2_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar2['value'])))
bar3_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar3['value'])))
bar4_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar4['value'])))
bar5_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar5['value'])))
bar6_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar6['value'])))
bar7_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar7['value'])))
bar8_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar8['value'])))
bar9_value = tkinter.Label(right_frame, text = ('%5.4f'%(bar9['value'])))

bar0_value.grid(column = 0, row = 4)
bar1_value.grid(column = 1, row = 4)
bar2_value.grid(column = 2, row = 4)
bar3_value.grid(column = 3, row = 4)
bar4_value.grid(column = 4, row = 4)
bar5_value.grid(column = 5, row = 4)
bar6_value.grid(column = 6, row = 4)
bar7_value.grid(column = 7, row = 4)
bar8_value.grid(column = 8, row = 4)
bar9_value.grid(column = 9, row = 4)

label_0 = tkinter.Label(right_frame, text = '0', font=('TkDefaultFont', 12, 'bold'))
label_1 = tkinter.Label(right_frame, text = '1', font=('TkDefaultFont', 12, 'bold'))
label_2 = tkinter.Label(right_frame, text = '2', font=('TkDefaultFont', 12, 'bold'))
label_3 = tkinter.Label(right_frame, text = '3', font=('TkDefaultFont', 12, 'bold'))
label_4 = tkinter.Label(right_frame, text = '4', font=('TkDefaultFont', 12, 'bold'))
label_5 = tkinter.Label(right_frame, text = '5', font=('TkDefaultFont', 12, 'bold'))
label_6 = tkinter.Label(right_frame, text = '6', font=('TkDefaultFont', 12, 'bold'))
label_7 = tkinter.Label(right_frame, text = '7', font=('TkDefaultFont', 12, 'bold'))
label_8 = tkinter.Label(right_frame, text = '8', font=('TkDefaultFont', 12, 'bold'))
label_9 = tkinter.Label(right_frame, text = '9', font=('TkDefaultFont', 12, 'bold'))

label_0.grid(column = 0, row = 6)
label_1.grid(column = 1, row = 6)
label_2.grid(column = 2, row = 6)
label_3.grid(column = 3, row = 6)
label_4.grid(column = 4, row = 6)
label_5.grid(column = 5, row = 6)
label_6.grid(column = 6, row = 6)
label_7.grid(column = 7, row = 6)
label_8.grid(column = 8, row = 6)
label_9.grid(column = 9, row = 6)

# Events for drawing on the canvas and creating our image to be analysed
white = np.float32(0.)
image1 = PIL.Image.new("F", (width, height), white)
draw = ImageDraw.Draw(image1)

# Binding events to canvas
canvas.bind("<B1-Motion>", paint)


### Run GUI
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mainWindow.mainloop()

