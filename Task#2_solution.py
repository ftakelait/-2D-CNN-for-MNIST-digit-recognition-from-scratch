#!/usr/bin/env python
# coding: utf-8

# # 2D CNN for MNIST digit recognition from scratch

# <img src="images/mnist_sample.png" style="width:30%">

# In[1]:


## Getting reproducible results
# How do I get stable results for trained NN? for this we need to seed a random number generator
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import tensorflow as tf  # !pip install tensorflow
import numpy as np
import warnings
import sys
from PIL import Image

warnings.filterwarnings('ignore')  ## Never print matching warnings

tf.logging.set_verbosity(tf.logging.ERROR)  ## Control logging by filtering out ERROR logs

print("We're using TF", tf.__version__)
print("We're using Python", sys.version)


# # Looking at the data
# 
# In this task we have 55000 28x28 images of digits from 0 to 9.
# We will train a classifier on this data.

# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X_train = mnist.train.images  ## numpy array of shape (55000, 784) i.e pixels pixels [0,1.0]
y_train = mnist.train.labels  ## numpy array of shape (55000, 10) i.e. one-hot encoded labels
# one-hot coding have 1 at corresponding position and zeros else
# examples:
#  0 is (1, 0, 0, 0, 0, 0, 0, 0, 0, 0,)
#  1 is (0, 1, 0, 0, 0, 0, 0, 0, 0, 0,)
#  2 is (0, 0, 1, 0, 0, 0, 0, 0, 0, 0,)
# ...
#  9 is (0, 0, 0, 0, 0, 0, 0, 0, 0, 1,)

X_test = mnist.test.images
y_test = mnist.test.labels


# In[4]:


# Images are already flattened which means that our linear model implementation now is simplified
print("X_train [shape %s]  whole sample:" % (str(X_train.shape)))
def show_img(sample):
    pixels = np.array(sample, dtype = 'float32')
    pixels = pixels.reshape((28,28))
    plt.imshow(pixels)
    plt.show()
show_img(X_train[1, :])
print("y_train [shape %s] 1 samples:\n" % (str(y_train.shape)), y_train[1])


# # Logistic regression with 2D CNN
# 
# Here we will train a linear classifier $\vec{x} \rightarrow y$ with SGD using low-level TensorFlow.
# 
# First, we need to calculate a logit (a linear transformation) $z_k$ for each class: 
# $$z_k = \vec{x} \cdot \vec{w_k} + b_k \quad k = 0..9$$
# ```python
# logits = X @ W1 + b1  ### logits for input_X, resulting shape should be [input_X.shape[0], 256]
# ```
# Then, we transform these logits $z_k$ to a valid probabilities $p_k$ with softmax: 
# $$p_k = \frac{e^{z_k}}{\sum_{i=0}^{9}{e^{z_i}}} \quad k = 0..9$$
# ```python
# def softmax(z):  ## this approach provides numerical stability
#     """Compute softmax values for each sets of scores in z."""
#     e = tf.exp(z - tf.reduce_max(z))
#     return e / tf.reduce_sum(e)
# ```
# In order to avoid numerical overflow we use numerically stable sigmoid function:
# $$p_k=\frac{1}{1+e^{-z_k}}=\frac{e^{z_k}}{1 + e^{z_k}} \quad k = 0..9$$
# where the distributions $[z_k,0]$ and $[0,−z_k]$ are equivalent.
# ```python
# def sigmoid(z):
#     """Numerically stable sigmoid function."""
#     # if z is less than zero then z will be small, denom can't be
#     # zero because it's 1+z.
#     return tf.where(z >= 0, 1 / (1 + tf.exp(-z)), tf.exp(z) / (1 + tf.exp(z))) 
# ```
# 
# Finally, We use a cross-entropy loss to train our multi-class classifier:
# $$\text{cross-entropy}(y, p) = -\sum_{k=0}^{9}{\log(p_k)[y = k]}$$ 
# 
# where 
# $$
# [x]=\begin{cases}
#        1, \quad \text{if $x$ is true} \\
#        0, \quad \text{otherwise}
#     \end{cases}
# $$
# 
# Cross-entropy minimization pushes $p_k$ close to 1 when $y = k$, which is what we want.
# ```python
# # The loss should be a scalar number: average loss over all the objects with tf.reduce_mean().
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(softmax(y_pred)), reduction_indices=[1]))
# ```
# Here's our plan:
# * We use a matrix placeholder for flattened `X_train` (28x28 = 784)
# * Convert `y_train` to one-hot (already done) encoded vectors that are needed for cross-entropy
# * We use a shared variable `W` for all weights (a column $\vec{w_k}$ per class) and `b` for all biases.
# * Our aim is: $\text{test accuracy} \geq 0.95$

# In[5]:


## Learning steps:
# 1. Get input (features) and true output (target)
# Placeholders for the input data
# first shape is None means that the batch size will be specified at runtime
X = tf.placeholder(tf.float64, shape=(None, 784), name='X')
y = tf.placeholder(tf.float64, shape=(None, 10), name='y')


# In[6]:


### Model parameters: W and b
# The initializer Xavier Glorot and Yoshua Bengio (2010) is designed to keep the 
# Scale of the gradients roughly the same in all layers
# Zeros_initializer is used for initializing the bias term with zero

# 1st hidden layer
W1 = tf.get_variable("W1", shape=(784, 256), dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=(256), dtype=tf.float64, initializer = tf.zeros_initializer)

# 2nd hidden layer
W2 = tf.get_variable("W2", shape=(256, 10), dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=(10), dtype=tf.float64, initializer = tf.zeros_initializer)


def sigmoid(z):
    """Numerically stable sigmoid function."""
    return tf.where(z >= 0, 1 / (1 + tf.exp(-z)), tf.exp(z) / (1 + tf.exp(z)))

def softmax(z):  ## this approach provides numerical stability
    """Compute softmax values for each sets of scores in z."""
    e = tf.exp(z - tf.reduce_max(z))
    return e / tf.reduce_sum(e)

# 2. Compute the "guess" (predictions) based on the features and weights
### Compute predictions
logits = X @ W1 + b1  ### Logits for input_X, resulting shape should be [input_X.shape[0], 256]
probas = sigmoid(logits)  ### Apply softmax to logits
y_pred = probas @ W2 + b2  ### Logits for probabilities, resulting shape should be [256, 10]

# 3. Compute the loss based on the difference between the predictions and the target
### Cross-Entropy loss 
# Hint: use tf.reduce_mean, tf.reduce_sum for reduction by specified axis
# The loss should be a scalar number: average loss over all the objects with tf.reduce_mean().
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(softmax(y_pred)), reduction_indices=[1]))

# 4. Update the weights (parameters) based on the gradient descent of the loss
# We use a default tf.train.AdamOptimizer to get an SGD step
lr = 0.01
optimizer = tf.train.AdamOptimizer(lr)
step = optimizer.minimize(cost)  ### optimizer step that minimizes the loss (cost)


# In[7]:


grads = tf.gradients(cost, tf.trainable_variables()) # take gradient of ALL TRAINABLES variables
grad_vars = list(zip(grads, tf.trainable_variables()))

# update = optimizer.apply_gradients(grads_and_vars=grad_list) # don't rename this op
capped_grads_vars = [[tf.clip_by_value(g, -5, 5), v] for g, v in grad_vars]
update = optimizer.apply_gradients(grads_and_vars=capped_grads_vars) # don't rename this op

### Apply tf.argmax to find a class index with highest probability
pred_classes = tf.argmax(y_pred, axis=1)  ## class index with highest probability for the predictions
true_classes = tf.argmax(y, axis=1)  ## class index with highest probability for the true outcome
pred = tf.cast(tf.equal(pred_classes, true_classes), dtype=tf.float32)
acc = tf.reduce_mean(pred) # don't rename this op


# ## Training

# In[9]:


BATCH_SIZE = 512  ## Number of samples that will be propagated through the network
EPOCHS = 120 # 1 Epoch is view of all elements in dataset 1 time
BATCH_NUM = int(X_train.shape[0]/BATCH_SIZE)  # Here you loose the rest of batch you can add it

# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()
sess = tf.Session()  ## Launch the graph in a session.
sess.run(init_op)  ## # Run the Op that initializes global variables.

for epoch in range(EPOCHS):  # we finish an epoch when we've looked at all training samples
    err = 0
    for batch in range(BATCH_NUM):
        # extract a batch
        batch_X = X_train[batch * BATCH_SIZE: batch * BATCH_SIZE + BATCH_SIZE]
        batch_y = y_train[batch * BATCH_SIZE: batch * BATCH_SIZE + BATCH_SIZE]

        # a key operation. Run 'operations' and place data in placeholders 'X' and 'y'
        _, c = sess.run([update, cost], feed_dict={X: batch_X, y: batch_y})
        err += c
    err /= BATCH_NUM
    if epoch % 20 == 9:
        # print(sess.run([logits, probas], feed_dict={X: batch_X, y: batch_y}))
        print("Test cost after %d epochs: %.4f" % (epoch + 1, err))

print("\nOPTIMIZATION IS DONE!")
acc_train = sess.run([acc], feed_dict={X: X_train, y: y_train})
print('Score = %f' % acc_train[0])


# Final output of accuracy is  99.985%.

# ## Testing

# In[9]:


img = np.invert(Image.open("images/test_img.png").convert('L')).ravel()


# In[10]:


prediction = sess.run(tf.argmax(y_pred, 1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))


# In[ ]:




