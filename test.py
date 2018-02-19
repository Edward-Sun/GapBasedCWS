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
from config import TRAIN_DATA_UNI, DEV_DATA_UNI, TEST_DATA_UNI, TEST_WHO
from config import TRAIN_DATA_BI, DEV_DATA_BI, TEST_DATA_BI, MEMORY
from config import WINDOW_SIZE, BIGRAM, UNIGRAM_DIM, UNIGRAM

# ==================================================
print('Generate words and characters need to be trained')
VOCABS = Vocab()
TAGS = Tag()
uni_embedding = VOCABS.uni_vectors
bi_embedding = VOCABS.bi_vectors
da_idx = Data_index(VOCABS, TAGS)
da_idx.process_all_data()

# model names
tf.flags.DEFINE_string("model_name", "cws_"+TASK_NAME, "model name")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Load data
print("Loading data...")
test_file = TEST_DATA_BI
    
test_df = pd.read_csv(test_file)
test_data_iterator = data_helpers.BucketedDataIterator(test_df)


final_score = None

num_of_test = 0

for WHO in TEST_WHO:

	num_of_test += 1

    with tf.Graph().as_default():
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        with sess.as_default():

            # build model
            model = Model(uni_embedding=uni_embedding,
                          bi_embedding=bi_embedding)

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

            filename =  str(UNIGRAM_DIM) + '_dense'\
            + '_' + str(WINDOW_SIZE) + '_' + str(BIGRAM) + '_' + str(UNIGRAM) + str(WHO)

            checkpoint_prefix = os.path.join(checkpoint_dir, filename)

            def evaluate_word_PRF(y_pred, y, test = False):
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
                        num_dict[i-start] += 1
                        if flag == True:
                            cor_num += 1
                            true_dict[i-start] += 1
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

                print(P,R,F)

            def final_test_step(df, iterator):
                N = df.shape[0]
                y_true, y_pred, y_score = model.fast_all_predict(sess, N, iterator)
                return y_pred, y_true, y_score

            saver.restore(sess, checkpoint_prefix)
            yp, yt, ys = final_test_step(test_df, test_data_iterator)
            evaluate_word_PRF(yp, yt)

            f_yt = open(Y_TRUE, 'w')
            f_yt.write(''.join(str(e) for e in yt))
            f_yt.close()

            f_ys = open(Y_SCORE + '_' + str(BIGRAM) + '_' + str(UNIGRAM) + str(WHO), 'w')
            f_ys.write(','.join(str(e) for e in ys))
            f_ys.close()

            if final_score is None:
                final_score = f_ys
            else:
                final_score = [(a * (num_of_test-1) + b)/num_of_test for a, b in zip(final_score, f_ys)]

if final_score is not None:

    yp = [0 if _ < 0.5 else 1 for _ in final_score]

    f_yp = open(Y_PRED, 'w')
    f_yp.write(''.join(str(e) for e in yp))
    f_yp.close()

    tmpOOV = OOV()
    oovrate = tmpOOV.eval_oov_rate()
    print('--------------Test ends-------------')