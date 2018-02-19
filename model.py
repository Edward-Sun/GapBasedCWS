import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, UNIGRAM_DIM, LEARNING_RATE0, LEARNING_RATE1, LEARNING_RATE2, WINDOW_SIZE, LAYER_NORM
from config import GREEDY, LR_TIME0, LR_TIME1, GREEDY_LOSS, INIT_DIM, UNIGRAM, WEIGHT_DECAY
from config import UNI_TRAINABLE, LABEL_SMOOTHING, GROWTH_RATE, LAYER, DENSE_BLOCK
from config import BIGRAM_DIM, BIGRAM, REPRESENTATION_DROPOUT_RATE, ACTIVATION


class Model(object):
    def __init__(self,
                 clip=1,
                 uni_embedding=None,
                 bi_embedding=None):
        self.clip = clip
        self.uni_embedding = uni_embedding
        self.bi_embedding = bi_embedding

        oov_vector = [np.zeros(UNIGRAM_DIM, dtype=np.float32)]
        oov_vector = np.asarray(oov_vector, dtype=np.float32)

        # placeholders        
        self.bi_x = tf.placeholder(tf.float32, [None, None, BIGRAM_DIM], name="bi_x")
        self.x = tf.placeholder(tf.int32, [None, None], name="x")
        self.y = tf.placeholder(tf.int32, [None, None], name="y")
        self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")     
        self.test_P = tf.placeholder(tf.float32, shape=(), name="test_P")
        self.test_R = tf.placeholder(tf.float32, shape=(), name="test_R")
        self.test_F = tf.placeholder(tf.float32, shape=(), name="test_F")
        self.dev_P = tf.placeholder(tf.float32, shape=(), name="dev_P")
        self.dev_R = tf.placeholder(tf.float32, shape=(), name="dev_R")
        self.dev_F = tf.placeholder(tf.float32, shape=(), name="dev_F")
        self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
        with tf.variable_scope("Weights"):
            weights = tf.cast(tf.reshape(tf.sequence_mask(self.seq_len), [-1]), tf.float32)
         
        with tf.variable_scope("Embedding"):
            if self.uni_embedding is None:
                self.uni_embedding = np.zeros([100, UNIGRAM_DIM], dtype=np.float32)
                        
            self.OOV_embedding = tf.Variable(
                oov_vector,
                name="OOV_embedding",
                trainable=False)
            self.uni_embedding = tf.Variable(
                self.uni_embedding[1:],
                name="uni_embedding",
                trainable=UNI_TRAINABLE)

            self.all_embedding = tf.concat([self.OOV_embedding, self.uni_embedding], 0)
            
            x = tf.nn.embedding_lookup(self.all_embedding, self.x, name="x_embedding")

            bi_x = self.bi_x

        with tf.variable_scope("Network_Architecture"):
            self.unary_scores = self.embedding2unary_scores(x, bi_x)           
            unary_scores = tf.reshape(self.unary_scores, [-1, 2]) 
        with tf.variable_scope("Loss"):
            with tf.variable_scope("Greedy"):
            # Greedy log Likelihood losses
                y = tf.reshape(tf.one_hot(self.y, 2), [-1, 2])
                
                y = y * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING/2
                
                greedy_loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=unary_scores,
                                                                weights=weights)        

            reg_loss =  tf.reduce_sum(tf.losses.get_regularization_losses())

            loss = reg_loss + greedy_loss

        with tf.variable_scope("loss") as scope:
            tf.summary.scalar('loss_greedy', greedy_loss)
            tf.summary.scalar('regularization', reg_loss)

        with tf.variable_scope("accuracy") as scope:
            tf.summary.scalar('test_P', self.test_P)
            tf.summary.scalar('test_R', self.test_R)
            tf.summary.scalar('test_F', self.test_F)
            tf.summary.scalar('dev_P', self.dev_P)
            tf.summary.scalar('dev_R', self.dev_R)
            tf.summary.scalar('dev_F', self.dev_F)

        with tf.variable_scope("train_ops") as scope:                    
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optimizer0 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE0)
                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE1)
                self.optimizer2 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE2)                   
                
                tvars = tf.trainable_variables()

                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)                
                                    
                self.train_op0 = self.optimizer0.apply_gradients(zip(grads, tvars),
                                                               global_step=self.global_step)
                self.train_op1 = self.optimizer1.apply_gradients(zip(grads, tvars),
                                                               global_step=self.global_step)
                self.train_op2 = self.optimizer2.apply_gradients(zip(grads, tvars),
                                                               global_step=self.global_step)
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.summary.merge_all()
    
    def shannon_entropy(self, unary_scores, weights):
        unary_softmax = tf.nn.softmax(unary_scores)                  
        the_shannon_entropy = - tf.reduce_sum(unary_softmax * tf.nn.log_softmax(unary_scores), 1)
        num_labels = tf.reduce_sum(weights)
        num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)
        return tf.reduce_sum(weights * the_shannon_entropy) / num_labels
    
    def embedding2unary_scores(self, x, bi_x = None):
        with tf.variable_scope("Emebedding_Dropout"):
            if UNIGRAM:
                x = tf.concat([x[:,:-1,:], x[:,1:,:]], axis = 2)
                if BIGRAM:
                    bi_x = bi_x[:,1:-1,:]
                    x = tf.concat([x, bi_x], axis = 2)       
            else:
                x = bi_x[:,1:-1,:]
        with tf.variable_scope("Init_Conv"):
            x = self.activate(x, INIT_DIM, 1)            
            x = tf.layers.dropout(x, rate = REPRESENTATION_DROPOUT_RATE, training = self.is_training)           
        if DENSE_BLOCK:
            with tf.variable_scope("Dense_Block"):                   
                for layer in range(LAYER):
                    with tf.variable_scope("Dense_Layer" + str(layer)):                                                          
                        x = self.bottleneck_dense_1d(x, GROWTH_RATE, 3)
        else:
            with tf.variable_scope("Residual_Block"):                   
                for layer in range(LAYER):
                    with tf.variable_scope("Residual_Layer" + str(layer)):                                                          
                        x = self.conv_1d(x, INIT_DIM, 3)
                    
        with tf.variable_scope("Softmax"):
            x = x[:,WINDOW_SIZE:-WINDOW_SIZE,:]
            
            unary_scores = tf.layers.conv1d(
                inputs=x, filters=2, 
                kernel_size=1, padding="same",
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        return unary_scores
    
    def lrelu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
    
    def activate(self, x, dim, kernal_size):      
        if ACTIVATION == 'GLU':
            x1 = tf.layers.conv1d(
                inputs=x, filters = dim, kernel_size=kernal_size,
                padding = 'same', activation=None, use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer())      
            _x1 = tf.layers.conv1d(
                inputs=x, filters = dim, kernel_size=kernal_size,
                padding = 'same', activation=None, use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer())         
            x = self.norm(x1) * tf.sigmoid(_x1)
        else:
            x = tf.layers.conv1d(
                inputs=x, filters = dim, kernel_size=kernal_size,
                padding = 'same', activation=None, use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer())            
            x = tf.nn.elu(self.norm(x))        
        return x
    
    def norm(self, inputs):
        if LAYER_NORM:
            return tf.contrib.layers.layer_norm(inputs)
        else:
            return tf.layers.batch_normalization(inputs, training=self.is_training)
    
    def dense_1d(self, x, growth_rate, kernal_size):
        conv = self.activate(x, growth_rate, kernal_size)
        return tf.concat([x, conv], axis = 2)
    
    def bottleneck_dense_1d(self, x, growth_rate, kernal_size):        
        conv = self.activate(x, growth_rate, 1)             
        conv = self.activate(conv, growth_rate, kernal_size)
        return tf.concat([x, conv], axis = 2)
    
    def conv_1d(self, x, output_dim, kernal_size):
        conv = self.activate(x, output_dim, kernal_size)
        conv = self.activate(conv, output_dim, kernal_size)   
        return x + conv
    
    def train_step(self, sess, bi_x_batch, x_batch, y_batch, seq_len_batch,
                   test_P, test_R, test_F, dev_P, dev_R, dev_F, dev = False):
        
        feed_dict = {
            self.bi_x: self.bi_embedding[bi_x_batch],
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len: seq_len_batch,
            self.test_P: test_P,
            self.test_R: test_R,
            self.test_F: test_F,
            self.dev_P: dev_P,
            self.dev_R: dev_R,
            self.dev_F: dev_F,
            self.is_training: True
        }
        
        my_step = sess.run(self.global_step)
        
        if my_step < LR_TIME0:
            train_op = self.train_op0           
        elif my_step < LR_TIME1:
            train_op = self.train_op1
        else:
            train_op = self.train_op2
            
        if dev:
            train_op = self.global_step
            
        _, step, summary = sess.run(
            [train_op, self.global_step, self.merged],
            feed_dict)
        
        if dev:
            return summary
        else:
            return step, summary

    def fast_all_predict(self, sess, N, batch_iterator):
        y_score, y_pred, y_true = [], [], []
        num_batches = int((N - 1) / BATCH_SIZE)

        for i in range(num_batches + 1):
            
            if i == num_batches:
                bi_x_batch, x_batch, y_batch, seq_len_batch = batch_iterator.rest_all_batch()
            else:
                bi_x_batch, x_batch, y_batch, seq_len_batch = batch_iterator.next_all_batch()
                
            # infer predictions
            feed_dict = {
                self.bi_x: self.bi_embedding[bi_x_batch],
                self.x: x_batch,
                self.seq_len: seq_len_batch
            }

            unary_scores= sess.run(self.unary_scores, feed_dict)

            for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
                # remove padding
                unary_scores_ = unary_scores_[:seq_len_]

                # Compute the highest scoring sequence.
                viterbi_sequence = [(0 if _[0] > _[1] else 1) for _ in unary_scores_]
                viterbi_sequence.append(1)
                score_sequence = [(np.exp(_) / np.sum(np.exp(_), axis=0))[1] for _ in unary_scores_]
                score_sequence.append(1)
                _y_true = y_[:seq_len_].tolist()
                _y_true.append(1)
                y_pred += viterbi_sequence
                y_score += score_sequence
                y_true += _y_true
        return y_true, y_pred, y_score

def test():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            model = Model()
    print("test over")


if __name__ == "__main__":
    test()
