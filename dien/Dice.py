import tensorflow as tf

def dice(_x, axis=1, epsilon=1e-6, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable("alpha"+name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())
        reduction_axis = list(range(len(input_shape)))
        del reduction_axis[axis]
        bordcast_axis = [1] * len(input_shape)
        bordcast_axis[axis] = reduction_axis[axis]

    mean = tf.reduce_mean(_x, axis=reduction_axis)
    bordcast_mean = tf.reshape(mean, bordcast_axis)
    std = tf.reduce_mean(tf.square(_x - bordcast_mean) + epsilon, axis=reduction_axis)
    std = tf.sqrt(std)
    bordcast_std = tf.reshape(std, bordcast_axis)
    x_normed = (_x - bordcast_mean) / (bordcast_std+epsilon)
    x_p = tf.sigmoid(x_normed)
    return alphas*(1.0 - x_p)*_x + x_p*_x


def parametic_relu(_x):
    alphas = tf.get_variable("alpha", _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.sigmoid(_x)
    neg = alphas*(_x - abs(_x))*0.5
    return pos + neg