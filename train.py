import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from voc import Vocab, OOV, Tag
from model import Model
import data_helpers
import logging
from prepare_data_index import Data_index
from collections import defaultdict

from config import TASK_NAME, LOG_PATH, MODEL_DIR, Y_PRED, Y_TRUE, Y_SCORE
from config import TRAIN_DATA_UNI, DEV_DATA_UNI, TEST_DATA_UNI
from config import TRAIN_DATA_BI, DEV_DATA_BI, TEST_DATA_BI, MEMORY
from config import WINDOW_SIZE, UNIGRAM_DIM, BIGRAM, UNIGRAM, DEVICE, MAX_STEP

# ==================================================
print('Generate words and characters need to be trained')
VOCABS = Vocab()
TAGS = Tag()
uni_embedding = VOCABS.uni_vectors
bi_embedding = VOCABS.bi_vectors

# model names
tf.flags.DEFINE_string("model_name", "cws_"+TASK_NAME, "model name")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", MAX_STEP, "Number of training steps")
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps and save")
tf.flags.DEFINE_integer("early_stop", 40000, "Early stop steps")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log pl:acement of ops on devices")

# TensorBoard Log

tf.flags.DEFINE_string("summaries_dir", LOG_PATH,"summaries directory")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

logger = logging.getLogger('record_base')
hdlr = logging.FileHandler('Baseline_train.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY))
    session_conf.gpu_options.visible_device_list= DEVICE
    #session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # build model
        model = Model(uni_embedding=uni_embedding,
                      bi_embedding=bi_embedding)
        
        #da_idx = Data_index(VOCABS, TAGS)
        #da_idx.process_all_data()

        # Load data
        print("Loading data...")
        train_file = TRAIN_DATA_BI
        dev_file = DEV_DATA_BI
        test_file = TEST_DATA_BI

        train_df = pd.read_csv(train_file)
        train_data_iterator = data_helpers.BucketedDataIterator(train_df)

        dev_df = pd.read_csv(dev_file)
        dev_data_iterator = data_helpers.BucketedDataIterator(dev_df)

        test_df = pd.read_csv(test_file)
        test_data_iterator = data_helpers.BucketedDataIterator(test_df)

        # Output directory for models
        out_dir = os.path.join(MODEL_DIR, FLAGS.model_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.join(out_dir, "checkpoints")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        saver = tf.train.Saver(tf.global_variables())
        
        if not os.path.exists(FLAGS.summaries_dir):
            os.makedirs(FLAGS.summaries_dir)
            
        #saver.restore(sess, checkpoint_prefix)
        summary_count = 0
        filename =  str(UNIGRAM_DIM) + '_dense'\
        + '_' + str(WINDOW_SIZE) + '_' + str(BIGRAM) + '_' + str(UNIGRAM) + str(summary_count)    

        train_summary_path = os.path.join(FLAGS.summaries_dir,
                                          "train" + filename)
        while os.path.exists(train_summary_path):
            summary_count += 1
            filename = str(UNIGRAM_DIM) + '_dense'\
            + '_' + str(WINDOW_SIZE) + '_' + str(BIGRAM) + '_' + str(UNIGRAM) + str(summary_count)
            train_summary_path = os.path.join(FLAGS.summaries_dir,
                                              "train" + filename)
            
        train_writer = tf.summary.FileWriter(train_summary_path, sess.graph)
        
        dev_summary_path = os.path.join(FLAGS.summaries_dir,
                                          "dev" + filename)
        
        dev_writer = tf.summary.FileWriter(dev_summary_path, sess.graph)
        
        checkpoint_prefix = os.path.join(checkpoint_dir, filename)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        step = 0
        test_P, test_R ,test_F = 0.93, 0.93, 0.93
        dev_P, dev_R ,dev_F = 0.93, 0.93, 0.93

        def train_step(bi_x_batch, x_batch, y_batch, seq_len_batch):
            global step
            step, summary = model.train_step(sess, bi_x_batch, 
                x_batch, y_batch, seq_len_batch, test_P, test_R ,test_F, dev_P, dev_R ,dev_F)

            time_str = datetime.datetime.now().isoformat()
            if step > 100:
                train_writer.add_summary(summary, step)

            return step
        
        def dev_loss(bi_x_batch, x_batch, y_batch, seq_len_batch):
            global step
            summary = model.train_step(sess, bi_x_batch, 
                x_batch, y_batch, seq_len_batch, test_P, test_R ,test_F, dev_P, dev_R ,dev_F, dev = True)

            time_str = datetime.datetime.now().isoformat()
            if step > 100:
                dev_writer.add_summary(summary, step)
        
            return
        
        def evaluate_word_PRF(y_pred, y, test = False):
            global test_P, test_R ,test_F
            global dev_P, dev_R ,dev_F
            cor_num = 0
            yp_wordnum = y_pred.count(1)
            yt_wordnum = y.count(1)
            true_dict = defaultdict(int)
            num_dict = defaultdict(int)
            start = 0
            for i in range(len(y)):
                if y[i] == 1:
                    flag = True
                    for j in range(start, i+1):
                        if y[j] != y_pred[j]:
                            flag = False
                    num_dict[i+1-start] += 1
                    if flag == True:
                        cor_num += 1
                        true_dict[i+1-start] += 1
                    start = i
                    
            if start != len(y)-1:
                flag = True
                for j in range(start, len(y)-1):
                    if y[j] != y_pred[j]:
                        flag = False
                num_dict[len(y) - 1 - start] += 1
                if flag == True:
                    cor_num += 1
                    true_dict[len(y) - 1 - start] += 1
                    
            print(num_dict)
            print(true_dict)
            print(len(y_pred))
            print(len(y))
            
            P = cor_num / float(yp_wordnum)
            R = cor_num / float(yt_wordnum)
            F = 2 * P * R / (P + R)
            if test:
                test_P, test_R ,test_F = P, R, F
                return P,R,F
            else:
                dev_P, dev_R ,dev_F = P, R, F
                return F

        def final_test_step(df, iterator):
            N = df.shape[0]
            y_true, y_pred, y_score = model.fast_all_predict(sess, N, iterator)
            return y_pred, y_true, y_score
        
        # train loop
        logger.info('Task_{} Training starts'.format(TASK_NAME))
        best_accuary = 0.0
        best_step = 0
        for i in range(FLAGS.num_epochs):
            bi_x_batch, x_batch, y_batch, seq_len_batch = train_data_iterator.next_batch()
            current_step = train_step(bi_x_batch, x_batch, y_batch, seq_len_batch)

            if current_step % 10 == 0:
                bi_x_batch, x_batch, y_batch, seq_len_batch = dev_data_iterator.next_batch()               
                dev_loss(bi_x_batch, x_batch, y_batch, seq_len_batch)
            
            if current_step % FLAGS.evaluate_every == 0:
                yp, yt, ys = final_test_step(dev_df, dev_data_iterator)
                tmpacc = evaluate_word_PRF(yp, yt)
                if best_accuary < tmpacc:
                    best_accuary = tmpacc
                    best_step = current_step
                    yp_test, yt_test, ys_test = final_test_step(test_df, test_data_iterator)
                    p, r, f = evaluate_word_PRF(yp_test, yt_test, test=True)
                    path = saver.save(sess, checkpoint_prefix)
                    print("Saved model checkpoint to {}\n".format(path))

                if current_step - best_step > FLAGS.early_stop:
                    print("Dev acc is not getting better in many steps, triggers normal early stop")
                    break

        #logger.info('-------------Show the results:{}--------------'.format(filename))
        #logger.info('P:{:.2f},R:{:.2f},F:{:.2f},step:{},'.format(100 * p, 100 * r, 100 * f, best_step))
        #saver.restore(sess, path)
        #yp, yt, ys = final_test_step(test_df, test_data_iterator)
        #evaluate_word_PRF(yp, yt, test=True)
        #f_yt = open(Y_TRUE, 'w')
        #f_yt.write(''.join(str(e) for e in yt))
        #f_yt.close()
        #f_yp = open(Y_PRED, 'w')
        #f_yp.write(''.join(str(e) for e in yp))
        #f_yp.close()
        #f_ys = open(Y_SCORE, 'w')
        #f_ys.write(''.join(str(e) for e in ys))
        #f_ys.close()
        
        #tmpOOV = OOV()
        #oovrate = tmpOOV.eval_oov_rate()
        #logger.info('OOV:{:.2f}\n'.format(100 * oovrate))
        #logger.info('--------------Train ends-------------')