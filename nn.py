import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = "layer%s"%(n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            Weight = tf.Variable(tf.random_uniform([in_size,out_size]))
            tf.summary.histogram(layer_name+"/weights",Weight)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.01)
            tf.summary.histogram(layer_name+"/biases",biases)

        with tf.name_scope("Ws_plus_b"):
            Ws_plus_b = tf.add(tf.matmul(inputs, Weight), biases)
        if activation_function is None:
            outputs = Ws_plus_b
        else:
            outputs = activation_function(Ws_plus_b)
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.1,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs, 1, 10, 1,activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, 2,activation_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]) )
    tf.summary.scalar("loss",loss)

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 ==0:
            print(i,sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
            result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
            writer.add_summary(result,i)