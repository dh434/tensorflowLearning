import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear, RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs


class VecAttGRUCell(RNNCell):
    def __init__(self, num_units, activation=None,
                 reuse=None, kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score):
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer == None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtypes)
            with vs.variable_scope("gates"):
                self._gate_linear = _linear(
                    [inputs, state],
                    2*self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer
                )
        value = math_ops.sigmoid(self._gate_linear([inputs,state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer
                )
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (1. - att_score) * state + att_score * c
        return new_h, new_h

class QAAttGRUCell(RNNCell):
  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(QAAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)

  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    new_h = (1. - att_score) * state + att_score * c
    return new_h, new_h


def prelu(_x, scope=""):
    with tf.name_scope(name=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def calc_auc(raw_arr):
    # raw_arr:[[pre,label]]
    arr = sorted(raw_arr, key=lambda x:x[0], reverse=True)
    pos, neg = 0.,0.
    for item in arr:
        if item[1] == 1.:
            pos += 1.
        else:
            neg += 1.

    fp,tp = 0.,0.
    xy_arr = []
    for item in arr:
        if item[1] == 1.:
            tp += 1.
        else:
            fp += 1.
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y+prev_y) / 2.)
            prev_x = x
            prev_y = y
    return auc

def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1,time_major=False, return_alphas=False):

    if isinstance(facts,tuple):
        facts = tf.concat(facts, axis=2)

    if time_major:
        facts = tf.transpose(facts, [1,0,2])

    mask = tf.equal(mask, tf.ones_like(mask))
    hidden_size = facts.get_shape().as_list()[-1]
    input_size = query.get_shape().as_list()[-1]

    w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.variable_scope("v"):
        tmp1 = tf.matmul(facts, w1) # [B, T, A]
        tmp2 = tf.tensordot(query, w2, axes=1) # [B,  A]
        tmp2 = tf.reshape(tmp2, [-1, 1, tf.shape(tmp2)[-1]]) # [B, 1, A]
        tmp = tf.tanh(tmp1 + tmp2) + b # [B, T, A]

    v_dot_tmp = tf.tensordot(tmp, v, axes=1, name = "v_dot_tmp") # [B, T]
    keys_mask = mask # [B, 1, T]
    padding = tf.ones_like(v_dot_tmp) *(-2**32+1)
    v_dot_tmp = tf.where(keys_mask, v_dot_tmp, padding)
    alphas = tf.nn.softmax(v_dot_tmp, name ="alphas")

    outputs = facts * tf.expand_dims(alphas,-1)
    outputs = tf.reshape(outputs, tf.shape(facts))
    if not return_alphas:
        return outputs
    else:
        return outputs, alphas

def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1,time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        facts = tf.concat(facts,2)
        query = tf.concat([query, query],axis=1)

    if time_major:
        facts = tf.transpose(facts,[1,0,2])

    mask = tf.equal(mask, tf.ones_like(mask))

    query_size = tf.shape(query)[-1]
    fact_size = tf.shape(facts)[-1]

    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name ="f1_att"+stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name ="f2_att"+stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name ="f3_att"+stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1,1,tf.shape(facts)[1]])

    #mask
    key_masks = tf.expand_dims(mask, 1)#[B,1,T]
    padding = tf.ones_like(d_layer_3_all) * (-2**32+1)
    scores = tf.where(key_masks, d_layer_3_all, padding)

    if softmax_stag:
        scores = tf.nn.softmax(scores)

    if mode == "SUM":
        output = tf.matmul(scores, facts)
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))

    return output

def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        facts = tf.concat(facts,2)
        query = tf.concat([query, query],axis=1)

    if time_major:
        facts = tf.transpose(facts,[1,0,2])

    mask = tf.equal(mask, tf.ones_like(mask))
    query_size = tf.shape(query)[-1]
    fact_size = tf.shape(facts)[-1]
    query = tf.layers.dense(query, fact_size, activation=None, name="f1_"+stag)
    query = prelu(query)

    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name="f1_att" + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name="f2_att" + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name="f3_att" + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    score = d_layer_3_all

    key_masks = tf.expand_dims(mask, 1)  # [B,1,T]
    padding = tf.ones_like(d_layer_3_all) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, d_layer_3_all, padding)


    if softmax_stag:
        scores = tf.nn.softmax(scores)

    if mode == "SUM":
        output = tf.matmul(scores, facts)
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))

    if return_alphas:
        return output, scores
    else:
        return output

def self_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch ,output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:,i,:], batch[:,0:i+1,:],ATTENTION_SIZE,mask[:,0:i+1,:],
                                               softmax_stag=True,stag=stag, mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(self_attention_tmp,i)
        return batch,output,i+1
    output_ta = tf.TensorArray(dtype=tf.float32,size=0,
                               dynamic_size=True,
                               element_shape=(facts[:,0,:].get_shape()))
    _,output_op,_ = tf.while_loop(cond, body,[facts,output_ta,0])
    self_attention = output_ta.stack()
    self_attention = tf.transpose(self_attention, perm=[1,0,2])
    return self_attention

def self_all_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch ,output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:,i,:], batch,ATTENTION_SIZE,mask,
                                               softmax_stag=True,stag=stag, mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(self_attention_tmp,i)
        return batch,output,i+1
    output_ta = tf.TensorArray(dtype=tf.float32,size=0,
                               dynamic_size=True,
                               element_shape=(facts[:,0,:].get_shape()))
    _,output_op,_ = tf.while_loop(cond, body,[facts,output_ta,0])
    self_attention = output_ta.stack()
    self_attention = tf.transpose(self_attention, perm=[1,0,2])
    return self_attention

def din_fcn_shine(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        facts = tf.concat(facts,2)
        query = tf.concat([query, query],axis=1)

    if time_major:
        facts = tf.transpose(facts,[1,0,2])

    mask = tf.equal(mask, tf.ones_like(mask))
    query_size = tf.shape(query)[-1]
    fact_size = tf.shape(facts)[-1]
    query = tf.layers.dense(query, fact_size, activation=None, name="f1_"+stag)
    query = prelu(query)

    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, fact_size, activation=tf.nn.sigmoid, name="f1_att" + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, fact_size, activation=tf.nn.sigmoid, name="f2_att" + stag)
    d_layer_2_all = tf.reshape(d_layer_2_all,tf.shape(facts))
    output = d_layer_2_all
    return output