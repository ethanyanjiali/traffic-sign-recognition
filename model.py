### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x18.
    conv1_W = tf.Variable(
        tf.truncated_normal(shape=(5, 5, 3, 18), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(18))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1],
                         padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x18. Output = 14x14x18.
    conv1 = tf.nn.max_pool(conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x48.
    conv2_W = tf.Variable(
        tf.truncated_normal(shape=(5, 5, 18, 48), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(48))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1],
                         padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x48. Output = 5x5x48.
    conv2 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

    # Flatten. Input = 5x5x48. Output = 1200.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 1200. Output = 360.
    fc1_W = tf.Variable(
        tf.truncated_normal(shape=(1200, 360), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(360))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 360. Output = 252.
    fc2_W = tf.Variable(
        tf.truncated_normal(shape=(360, 252), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(252))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(
        tf.truncated_normal(shape=(252, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits