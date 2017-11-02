# MLHandwriting
Classifying Handwritten Digits with TensorFlow Library while using the MNIST database! 

*The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.

Required dependencies - TensorFlow and Docker.
If you don't have TensorFlow , You can download it here : https://www.tensorflow.org/install/ 
If you don't have Docker , You can download it here : https://www.docker.com/get-docker 

Open the docker quickstart terminal and copy paste the IP address to a notepad. (We'll use this I.P address later) 
Next , we'll launch a docker container with a tensorflow image. 

The image is located here : https://goo.gl/8fmqVW
The image contains tensorflow, with it's all it's dependencies properly configured. 

Use the following command to download and launch the iamge : docker run -it -p 8888:8888 tensorflow/tensorflow:0.10.0rc0. 

Afterwards, open up a new browser window and point an empty tab to the following IP address we copied down previously. 

You'll see the python notebook, but remember to point it to port 8888. 

# Now We'll download and import the MNIST dataset! 
Use the code 

mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

This code displays an image along with it's label. 

def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)

Reshape the image as their low resolution (28 x 28) has 784 pixels so we have 784 features. 

To flatten an image we convert a 2d array to a 1d array by unstacking the rows and lining them up , that's why we have to reshape the array to display it earlier. 

# Initialize the classifier 
Use a linear classifier that would give us approx 90% accuracy. 
It will provide 2 parameters. 
The first (n_classes=10) indicates how many classes we have and there are 10 - One for each type of digit. 
The second (feature_columns) informs the classifier of the features we'll be using. 

The classifier adds up the evidence tha thte iamge is each type of digit. 
the input nodes are on the top , represented by X's and the output nodes are on the bottom repreented by Y's. 

we have one input node for each feature or pixel in the image. and one output node for each digit the image could represent. 

We have 784 inputs and 10 outputs. 28 x 28 = 784. 

The inputs and outputs are fully connected. 
Each of these edges have a weight. 
When we classify an image , each pixel takes a route. 

It first , flows through the input node, then travels along the edges. Along the way it's multiplied by the weight on the edge. 
The output node then gathers evidence that the iamge we're classifying represents each type of number. 

The more evidence we get, the more likely the accuracy of an image would be. 

To calculate how much evidence we have , we add up the value of the pixel intensities multiplied by the weights. 

We can then predict that the images belongs to the output node with the most evidence. 
The most important parts are the weights. 
By setting them properly , we can get accurate classifications! 

# Evaluate the model 
classifier.evaluate(test_data, test_labels)
print classifier.evaluate(test_data, test_labels)["accuracy"]


0.9141

You can see that it is 91% accurate! 


















