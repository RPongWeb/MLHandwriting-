# import dependencies 

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# import dataset 
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Number of examples , you may wish to limit the size for faster experimentation. 
max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]

# Display Digits
def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
    
    
display (1) 
display (2) 

print len(data[0]) #784

# Evaluate accuracy 
classifier.evaluate(test_data, test_labels)
print classifier.evaluate(test_data, test_labels)["accuracy"] 

#0.9141 

# Classify a few examples 

print ("Predicted %d, Label: %d" % (classifier.predict(test_data[0]), test_labels[0]))
display(0) 

Predicted 7 , Label : 7 













































































































