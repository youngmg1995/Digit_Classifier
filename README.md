# Digit_Classifier
### Create Neural Networks to Classify Handwritten Digits

This project is meant to serve as an introduction to neural networks through the simple use-case of classifying handwritten digits. To accomplish this we did the following: 
1) Built, trained, and tested simple fully connected neural networks from scratch using numpy and the MNIST database
2) Improved on our simple models by incorporating advanced techniques into our networks 
3) Leveraged the advanced ML library Tensorflow to seemlessly build more comprehensive models using our new understanding of ANNs
4) Constructed interactive GUIs to compare the networks in action and have fun classifying our own handwritten digits

## Included Files

### Network_Builder/
This directory contains all the files used to build, train, and test our neural network models. These versions of the files in this repository are the exact scripts used to create the networks used in the interactive GUIs. However, these files can easily be modified to create and test newly designed networks. Below is a list of all the files/directories included with descriptions to help one navigate the files.
- **Network_Builder.py**: Script that constructs and saves the two simple fully connected neural networks used in the interactive GUI - "Classifier GUI". These models are built using the classes constructed in "network" and does not require any adavanced ML libraries, just numpy. This file should be used in conjunction with "networks" for anyone who wants to begin learning how to construct simple ANNs from scratch and develop an understanding of the basic techniques used to do so (Ex: Cost Functions, Gradient Descent, Backpropogation, etc.).
- **Network_Builder.py**: Script that constructs and saves the three neural networks used in the interactive GUI - "Classifier GUI TF". These models are built using Tensorflow, an free, open-source ML library developed by Google. Because of this, you must use a python environment with Tensorflow and Keras to run this script. This file should serve as a good introduction to using Tensorflow to easily build neural networks. In particular it also allows users to begin working with convolutional neural networks and emphasises the improved performance and ease of construction that comes with a comprehensive, end-to-end ML platform like Tensorflow.
- **network.py**: This file constructs the classes that can be used to build simple neural networks from scratch. In particular it creates the "network" class which stores the weights and biases for each fully connected network we design and houses methods for training the network through gradient descent and backpropogation.
- **mnist_loader.py**: This file contains functions that are used to load and prepare the images of handwritten digits used to train and test our models. The data we use comes from the MNIST database, http://yann.lecun.com/exdb/mnist/, and is stored in the folder "MNIST_Data".
- **MNIST_Data/**: Folder containing the MNIST handwritten digit images under the GZ file "mnist.pkl.gz".
- **Saved_Networks**: Folder containing all the saved networks created using the "Network_Bulder" files and leveraged in the interactive GUIs. The simple networks contructed using the "network" modules are saved in the folder as json files while the Tensorflow networks are saved to there own directories that contain the variables and assets for each model.

## Classifier GUIs
What use is building a model for classifying digits if we can't use it name our own handwritten numbers? To have some fun using our models, we built and included two interactive GUIs built using tkinter to do just that. Each GUI allows us to choose which model classifies our handwritten digit and displays the results along with the output activations for each digit in 0-9. Below is a list of the files for the GUIs and all other files used in its contruction.
- **Classifier GUI.py**: Interactive GUI for drawing and classifying handwritten digits using the two models constructed with "Network_Builder". The first model is a small fully connected layer with shape 784-30-10 while the second is more optimized and has the much larger shape 784-800-800-10. See descriptions in "Network_Builder" file for more details on the specific structures of each network and the techniques/parameters used to train the models. This file does not require Tensorflow to run.
- **Classifier GUI TF.p**: Interactive GUI for drawing and classifying handwritten digits using the three models constructed with "Network_Builder_TF". The first model is a small fully connected layer with shape 784-30-10, the second is simlar but more optimized and has the much larger shape 784-800-800-10, and the third is a convolutional network with two convolution layers followed by 2 fully connected layers. See descriptions in "Network_Builder_TF" file for more details on the specific structures of each network and the techniques/parameters used to train the models. This file does not require Tensorflow to run.
- **Number_Images/**: Folder containinng images of standard digits 0-9. They are used to display the digit classification assigned to our canvas image in the interactive GUIs.
- **Saved_GUI_Images**: Folder used to save images from the interactive GUIs. Specifically the reduced 28 x 28 pixel versions of the images drawn on the canvas that our model evaluates.

## References and Contributers
<p>Firstly, I would like to thank Michael Nielson who inspired this project and provided much of the knowledge basis and code used to get me started. I encourage anyone who is interested in neural networks, liked this project, and wants to learn more to visit his websit at: http://neuralnetworksanddeeplearning.com/index.html. This website contains the free book I used to learn the basics of neural networks, building them in python, and applying them to this simple use-case of recognizing handwritten digits. Additionally, you can find his code used to build these neworks at his Github, https://github.com/mnielsen/neural-networks-and-deep-learning, or the following Github for python 3 code, https://github.com/MichalDanielDobrzanski/DeepLearningPython35. In particular, the "mnist_loader.py" code found here is the exact code from Michaels project and the "network.py" modules were heavily influenced by Michaels own network modules. I encourage anyone who visits his site and finds his resources helpful to donate to his project.</p>
<p>Additionally, I would like to thank Jeff Heaton at Washington Universty in St. Lious whose youtube video, https://www.youtube.com/watch?v=qrkEYf-YDyI, helped me set up my python environment with Tensorflow and Keras. Prior to finding his tutorial, I struggled for days to set up a working tensorflow environment on Windows 10 that could leverage my PC's GPU. I encourage anyone using windows who wants to set up a similar environment to watch his 30 minute tutorial and leverage https://www.tensorflow.org/ for learning the basics of tensorflow.</p>
<p>Lastly, I would like to thank the youtube channel 3Blue1Brown whose videos first got me interested in machine learning and neural networks. Use the following link, https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw, to visit his channel and watch his videos exploring various topics on mathematics and computer science.</p>
