# simple neural network for XOR  problem with tensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
XOR - Problem
Neural Network properties:
Hidden Layer: 1
Hidden Nodes: 5
Learning Rate: 0.1
Training steps: 500
Activation function: Sigmoid
Backprob: Adam Optimizer
'''

# Forward function
def forward(x, w1, b1, w2, b2, train=True):
    Z = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    Z2 = tf.matmul(Z, w2) + b2
    if train:
        return Z2
    return tf.nn.sigmoid(Z2)

# init weights and bias
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([[1], [1], [0], [0]])

# define placeholders for input x and output y

phX = tf.placeholder(tf.float32, [None, 2])
phY = tf.placeholder(tf.float32, [None, 1])

# init weights
# 5 hidden nodes
w1 = init_weights([2, 5])
b1 = init_weights([5])
w2 = init_weights([5, 1])
b2 = init_weights([1])

y_hat = forward(phX, w1, b1, w2, b2)
pred = forward(phX, w1, b1, w2, b2, False)

# init learning rate
lr = 0.1

# init epochs
epochs = 500

# init cost function
cost = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_hat, labels=phY))

# init train function witch adam optimizer
train = tf.train.AdamOptimizer(lr).minimize(cost)

# save costs for plottig
costs = []

# create session and init variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# start training
for i in range(epochs):
    sess.run(train, feed_dict={phX: x, phY: y})

    c = sess.run(cost, feed_dict={phX: x, phY: y})
    costs.append(c)

    if i % 100 == 0:
        print(f"Iteration {i} .Cost: {c}")

print("Training complete")

# Make prediction

prediction = sess.run(pred, feed_dict={phX:x})

print("Porcentages: ")
print(prediction)

print("Prediction: ")
print(np.round(prediction))


# plot cost
plt.plot(costs)
plt.show()