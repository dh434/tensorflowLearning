import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def gen_data(size= 100000):
    X = np.random.choice(2,(size,))
    Y = []
    for i in range(size):
        threshold = 0.5
        if i-3 >=0 and X[i-3]==1:
            threshold += 0.5
        if i-8 >=0 and X[i-8] ==1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X,np.array(Y)

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.float32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.float32)

    for i in range(batch_size):
        data_x[i] = raw_x[i*batch_partition_length:(i+1)*batch_partition_length]
        data_y[i] = raw_y[i*batch_partition_length:(i+1)*batch_partition_length]

    epoch_size = batch_partition_length // num_steps
    for step in range(epoch_size):
        x = data_x[:,step*num_steps:(step+1)*num_steps]
        y = data_y[:,step*num_steps:(step+1)*num_steps]
        yield (x,y)

def gen_epochs(n, num_step):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps=num_step)

batch_size = 5
num_steps = 5
state_size = 10
n_classes = 2
learning_ratio = 0.1


x = tf.placeholder(tf.int32, [batch_size, num_steps])
y = tf.placeholder(tf.int32, [batch_size, num_steps])

init_state = tf.zeros([batch_size, state_size])

x_one_hot = tf.one_hot(x,n_classes)#[batch_size, num_steps, num_class]

rnn_inputs = tf.unstack(x_one_hot,axis=1,num=num_steps)

with tf.variable_scope("rnn_cell"):
    W = tf.get_variable("W", [n_classes+state_size, state_size])
    b = tf.get_variable("b", [state_size], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
    with tf.variable_scope("rnn_cell",reuse=True):
        W = tf.get_variable("W", [n_classes + state_size, state_size])
        b = tf.get_variable("b", [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], axis=1), W)+b)

state = init_state
rnn_outputs = []

for rnn_input in  rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]

# cell = tf.contrib.rnn.BasicRNNCell(state_size)
# rnn_outputs,final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, init_state=init_state)
#
# rnn_inputs = x_one_hot
# rnn_outputs,final_state = tf.contrib.rnn.dynamic_rnn(cell, rnn_inputs, init_state=init_state)

with tf.variable_scope("sotfmax"):
    W = tf.get_variable("W", [state_size, n_classes])
    b = tf.get_variable("b", [n_classes])
logits = [ tf.matmul(rnn_output, W)+b for rnn_output in rnn_outputs]
predictions = [ tf.nn.softmax(logit) for logit in logits] #10,5,2

y_as_list = tf.unstack(y, num=num_steps, axis=1)


losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logit) for label,logit in zip(y_as_list,predictions)]
# losses = [tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit) for label, logit in zip(y_as_list, predictions)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_ratio).minimize(total_loss)

### 当使用dynamic_rnn
# logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs,[-1,state_size]),W)+b, [batch_size,num_steps,n_classes])
# predictions = tf.nn.softmax(logits)
# losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
# total_loss = tf.reduce_mean(losses)
# train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_ratio).minimize(total_loss)


def train_network(num_epochs, num_steps, state_size, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []



        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("Epoch: ",idx)
            for step, (X,Y) in enumerate(epoch):
            #     print(X.shape)
            #     temp,temp1,temp2 = sess.run([y_as_list,logits,predictions],
            #                          feed_dict={x: X, y: Y, init_state: training_state})
            #     print(np.array(temp).shape)
            #     print(np.array(temp1).shape)
            #     print(np.array(temp2).shape)
            #     break
            # break
                tr_losses, training_loss_, training_state,_ = sess.run([losses,total_loss, final_state,train_step],
                                                                       feed_dict={x:X,y:Y,init_state:training_state})
                training_loss += training_loss_
                if step%100 == 0 and step>0:
                    if verbose:
                        print("Average loss at step",step,
                              "for last 100 steps",training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0
        return training_losses

train_losses = train_network(1,num_steps,state_size)
plt.plot(train_losses)
plt.show()