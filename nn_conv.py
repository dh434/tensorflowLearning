import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def compute_accuracy(v_xs, v_ys, sess):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,axis=1), tf.argmax(v_ys, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1] ,strides=[1,2,2,1], padding="SAME")

def weight_variable(shape):
    intial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial)

def biases_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# batch_size, width, length, channel
x_image = tf.reshape(xs, [-1, 28,28,1])

## conv1 layer
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = biases_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer
w_conv2 = weight_variable([5,5,32,64])
b_conv1 = biases_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv1)
h_pool2 = max_pool_2x2(h_conv2)

## func1 layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = biases_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1  = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

## func2 layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels, sess), i)

