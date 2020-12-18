import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice,parametic_relu

class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM,  HIDDEN_SIZE, ATTENTION_SIZE, use_negSample):
        with tf.variable_scope("Inputs"):
            self.mid_his_batch_ph = tf.placeholder(tf.float32, [None,None], name="mid_his_batch_ph")
            self.cat_his_batch_ph = tf.placeholder(tf.float32, [None,None], name="cat_his_batch_ph")
            self.uid_batch_ph = tf.placeholder(tf.float32, [None,], name="uid_batch_ph")
            self.mid_batch_ph = tf.placeholder(tf.float32, [None,], name="mid_batch_ph")
            self.cat_batch_ph = tf.placeholder(tf.float32, [None,], name="cat_batch_ph")
            self.mask = tf.placeholder(tf.float32, [None,None], name="his_mask")
            self.seq_len_ph = tf.placeholder(tf.float32, [None,], name="seq_len_ph")
            self.target_ph = tf.placeholder(tf.float32, [None,None], name="target_ph")
            self.lr = tf.placeholder(tf.float64,[])
            self.use_negsampling = use_negSample
            if use_negSample:
                self.noclk_mid_batch_ph = tf.placeholder(tf.float32, [None, None, None], name="noclk_mid_batch_ph")
                self.noclk_cat_batch_ph = tf.placeholder(tf.float32, [None, None, None], name="noclk_cat_batch_ph")

        with tf.variable_scope("Embedding_layer"):
            self.uid_embedding_var = tf.get_variable("uid_embedding_var",[n_uid,EMBEDDING_DIM])
            tf.summary.histogram("uid_embedding_var", self.uid_embedding_var)
            self.uid_batch_embedding = tf.nn.embedding_lookup(self.uid_embedding_var, self.uid_batch_ph)

            self.mid_embedding_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram("mid_embedding_var", self.mid_embedding_var)
            self.mid_batch_embedding = tf.nn.embedding_lookup(self.mid_embedding_var, self.mid_batch_ph)
            self.mid_his_batch_embedding = tf.nn.embedding_lookup(self.mid_embedding_var, self.mid_his_batch_ph)
            if use_negSample:
                self.noclk_mid_batch_embedding = tf.nn.embedding_lookup(self.mid_embedding_var, self.noclk_mid_batch_ph)

            self.cat_embedding_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram("cat_embedding_var", self.cat_embedding_var)
            self.cat_batch_embedding = tf.nn.embedding_lookup(self.cat_embedding_var, self.cat_batch_ph)
            self.cat_his_batch_embedding = tf.nn.embedding_lookup(self.cat_embedding_var, self.cat_his_batch_ph)
            if use_negSample:
                self.noclk_cat_batch_embedding = tf.nn.embedding_lookup(self.cat_embedding_var, self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedding, self.cat_batch_embedding],1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedding, self.cat_his_batch_embedding], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if use_negSample:
            self.noclk_item_his_eb = tf.concat([self.noclk_mid_batch_embedding[:,:,0,:],
                                                self.noclk_cat_batch_embedding[:,:,0,:]], -1)
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb, [-1,tf.shape(self.noclk_mid_batch_embedding)[1],EMBEDDING_DIM])

            self.noclk_his_eb = tf.concat([self.noclk_mid_batch_embedding, self.noclk_cat_batch_embedding],-1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb,2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1,1)

    def build_fcn_net(self, imp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=imp, name="b1")
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name="f1")
        if use_dice:
            dnn1 = dice(dnn1, name="dice_1")
        else:
            dnn1 = prelu(dnn1, "prelu1")

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name="f2")
        if use_dice:
            dnn2 = dice(dnn2, name="dice_2")
        else:
            dnn2 = prelu(dnn2, "prelu2")

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name="f3")
        self.y_hat = tf.nn.softmax(dnn3) + 1e-6

        with tf.name_scope("Metric"):
            self.ctr_loss = tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = self.ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar("loss",self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar("accuracy",self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_state, click_seq, noclk_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input = tf.concat([h_state, click_seq], -1)
        noclk_input = tf.concat([h_state, noclk_seq], -1)
        click_prop_ = self.auxiliary_network(click_input)
        noclk_prop_ = self.auxiliary_network(noclk_input)

        click_loss = -tf.reshape(tf.log(click_prop_), [-1, click_seq.get_shape()[1]]) * mask
        noclk_loss = -tf.reshape(tf.log(noclk_prop_), [-1, noclk_seq.get_shape()[1]]) * mask
        loss_ = tf.reduce_mean(click_loss + noclk_loss)
        return loss_

    def auxiliary_network(self, in_, stag="auxiliary_net"):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                   feed_dict={
                                                       self.uid_batch_ph: inps[0],
                                                       self.mid_batch_ph: inps[1],
                                                       self.cat_batch_ph: inps[2],
                                                       self.mid_his_batch_ph: inps[3],
                                                       self.cat_his_batch_ph: inps[4],
                                                       self.mask: inps[5],
                                                       self.target_ph: inps[6],
                                                       self.seq_len_ph: inps[7],
                                                       self.lr: inps[8],
                                                       self.noclk_mid_batch_ph: inps[9],
                                                       self.noclk_cat_batch_ph: inps[10],
                                                   })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.noclk_mid_batch_ph: inps[8],
                self.noclk_cat_batch_ph: inps[9],
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7]
            })
            return probs, loss, accuracy, 0
        
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)
class Model_DIN(Model):
    def __init__(self,n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,use_negSample=False):
        super(Model_DIN,self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM,HIDDEN_SIZE, ATTENTION_SIZE,use_negSample)

        #Attention_layer
        with tf.name_scope("Attention_layer"):
            attention_output = din_attention(self.item_eb,self.item_his_eb,ATTENTION_SIZE,self.mask) #[B,T,H]
            att_fea = tf.reduce_sum(attention_output,1)
            tf.summary.histogram("att_fea",att_fea)
        inp = tf.concat([self.uid_batch_embedding, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, att_fea], axis=1)
        self.build_fcn_net(inp, use_dice=True)

class Model_DIN_V2_Gru_att_Gru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_att_Gru, self).__init__(n_uid, n_mid, n_cat,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        with tf.name_scope("rnn_1"):
            rnn_outputs,_ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                        sequence_length=self.seq_len_ph, dtype=tf.float32,
                                        scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope("Attention_layer_1"):
            attention_output, alpha = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                        softmax_stag=1, stag='1_1', mode="LIST",
                                                        return_alphas=True)
            tf.summary.histogram("alpha_outputs", alpha)

        with tf.name_scope("rnn_2"):
            rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.attention_output,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_2_Final_state', final_state2)

        inp = tf.concat([self.uid_batch_embedding, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, final_state2],1)
        self.build_fcn_net(inp, use_dice=True)

class Model_DIN_V2_Gru_Gru_att(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_Gru_att, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_outputs', rnn_outputs2)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs2, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, att_fea], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_DIN_V2_Gru_QA_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_QA_attGru, self).__init__(n_uid, n_mid, n_cat,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                         use_negsampling)

        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)

        with tf.name_scope("rnn_2"):
            rnn_outputs2, final_state2 = dynamic_rnn(QAAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                    att_scores=tf.expand_dims(alphas,-1),
                                                    sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                    scope="gru2")
            tf.summary.histogram('GRU2_Final_state', final_state2)

        inp = tf.concat([self.uid_batch_embedding, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, final_state2], axis=1)
        self.build_fcn_net(inp, use_dice=True)

class Model_DIN_V2_Gru_Vec_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_Vec_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        #inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
            aux_loss = self.auxiliary_loss(rnn_outputs[:,:-1,:], self.item_his_eb[:,1:,:],
                                           self.noclk_item_his_eb[:,1:,:],self.mask[:,1:], stag="gru")
            self.aux_loss = aux_loss

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        #inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)
