# Divya_S_(Projects)HAND WRITTEN DIGIT RECOGNIZER
It contains the project details and source code
Recognition is identifying or distinguishing a thing or an individual from the past experiences or learning. Similarly, Digit Recognition is nothing but recognizing or identifying the digits in any document. Digit recognition framework is simply the working of a machine to prepare itself or interpret the digits. Handwritten Digit Recognition is the capacity of a computer to interpret the manually written digits from various sources like messages, bank cheques, papers, pictures, and so forth and in various situations for web based handwriting recognition on PC tablets, identifying number plates of vehicles, handling bank cheques, digits entered in any forms etc. Machine Learning provides various methods through which human efforts can be reduced in recognizing the manually written digits. Deep Learning is a machine learning method that trains computers to do what easily falls into place for people: learning through examples. With the utilization of deep learning methods, human attempts can be diminished in perceiving, learning, recognizing and in a lot more regions. Using deep learning, the computer learns to carry out classification works from pictures or contents from any document. Deep Learning models can accomplish state-of-art accuracy, beyond the human level performance. The digit recognition model uses large datasets in order to recognize digits from distinctive sources. There are diverse challenges faced while attempting to solve this problem. The handwritten digits are not always of the same size, thickness, or orientation and position relative to the margins. The main objective was to actualize a pattern characterization method to perceive the handwritten digits provided in the MINIST data set of images of handwritten digits (0‐9).

3.1 ALGORITHM
	The CNN algorithm is used for creating an handwritten digit recognizer. The name “convolutional neural network” indicates that the network employs a mathematical operation called Convolution. 
	Convolution is a specialized kind of linear operation. Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

3.2 PACKAGE INSTALLATION
Installing the necessary libraries for this project using this command:
	Pip install numpy, tensorflow, keras, pillow

3.3 THE MNIST DATASET
	This is probably one of the most popular datasets among machine learning and deep learning enthusiasts. 
	The MNIST dataset contains 60,000 training images of handwritten digits from zero to nine and 10,000 images for testing. 
	So, the MNIST dataset has 10 different classes. The handwritten digits images are represented as a 28×28 matrix where each cell contains grayscale pixel value.

3.4 IMPORT THE LIBRARIES AND DATASET
	The first step is to import all the modules that we are going to need for training our model.

	The Keras library already contains some datasets and MNIST is one of them. 

	So we can easily import the dataset and start working with,The mnist.load_data() method returns us the training data, its labels and also the testing data and its labels.

3.5 PREPROCESS THE DATA 
	The image data cannot be fed directly into the model so we need to perform some operations and process the data to make it ready for our neural network. 
	The dimension of the training data is (60000,28,28). The CNN model will require one more dimension so we reshape the matrix to shape (60000,28,28,1).

3.6 CREAT THE MODEL
	Now we will create our CNN model in Python data science project. 
	A CNN model generally consists of convolutional and pooling layers.
	It works better for data that are represented as grid structures, this is the reason why CNN works well for image classification problems. 
	The dropout layer is used to deactivate some of the neurons and while training, it reduces offer fitting of the model.
	 We will then compile the model with the Adadelta optimizer.

3.7 TRAIN THE MODEL

	The model.fit() function of Keras will start the training of the model. 
	It takes the training data, validation data, epochs, and batch size.
	It takes some time to train the model. After training, we save the weights and model definition in the ‘mnist.h5’ file.

3.8 EVALUATE THE MODEL

	We have 10,000 images in our dataset which will be used to evaluate how good our model works. 
	The MNIST dataset is well balanced so we can get around 99%  accuracy.
3.9 CREATE THE GUI TO PREDICT DIGITS

	Now for the GUI, we have created a new file in which we build an interactive window to draw digits on canvas and with a button, we can recognize the digit. 
	The Tkinter library comes in the Python standard library,We have created a function predict_digit() that takes the image as input and then uses the trained model to predict the digit.
	Then we create the App class which is responsible for building the GUI for our app. We create a canvas where we can draw by capturing the mouse event and with a button, we trigger the predict_digit() function and display the results.
