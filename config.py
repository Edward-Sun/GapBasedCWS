# -*- coding: utf-8 -*-

import os

TASK_NAME = 'weibo'
MAX_STEP = 40001
TEST_WHO = [0, 1, 2, 3]
DEVICE = '0'
MEMORY = 0.25
BIGRAM = True
UNIGRAM = True
LEARNING_RATE0 = 0.002
LEARNING_RATE1 = 0.0002
LEARNING_RATE2 = 0.00002
LR_TIME0 = 6000
LR_TIME1 = 20000
LAYER_NORM = False
BATCH_SIZE = 64#256
WINDOW_SIZE = 3
REPRESENTATION_DROPOUT_RATE = 0.6
UNI_TRAINABLE = True
UNIGRAM_DIM = 50
BIGRAM_DIM = 50
INIT_DIM = 512
DENSE_BLOCK = True
LAYER = 12
GROWTH_RATE = 128
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.0
ACTIVATION = 'GLU'

DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, 'data_'+TASK_NAME)
MODEL_DIR = os.path.join(DIR, 'models')

UNIGRAM_PATH = os.path.join(MODEL_DIR, 'unigram' + str(UNIGRAM_DIM) + '.txt')
BIGRAM_PATH = os.path.join(MODEL_DIR, 'bigram' + str(BIGRAM_DIM) + '.txt')
GIGA_PATH = os.path.join(MODEL_DIR, 'giga')

DICT_PATH = os.path.join(DATADIR, 'words')
TRAIN_PATH = os.path.join(DATADIR, 'train')
LOG_PATH = os.path.join(DATADIR, 'log')
DEV_PATH = os.path.join(DATADIR, 'dev')
TEST_PATH = os.path.join(DATADIR, 'test')
GOLD_PATH = os.path.join(DATADIR, 'test_gold_edited')
DEST_PATH = os.path.join(DATADIR, 'dest')
Y_SCORE = os.path.join(DATADIR, 'y_score')
Y_PRED = os.path.join(DATADIR, 'y_pred')
Y_TRUE = os.path.join(DATADIR, 'y_true')

TRAIN_DATA_UNI = os.path.join(DATADIR, str(UNIGRAM_DIM)+ '_' + str(WINDOW_SIZE) + '_train_uni.csv')
DEV_DATA_UNI = os.path.join(DATADIR, str(UNIGRAM_DIM)+ '_' + str(WINDOW_SIZE) + '_dev_uni.csv')
TEST_DATA_UNI = os.path.join(DATADIR, str(UNIGRAM_DIM)+ '_' + str(WINDOW_SIZE) + '_test_uni.csv')

TRAIN_DATA_BI = os.path.join(DATADIR, str(UNIGRAM_DIM) + '_' + str(WINDOW_SIZE) + '_train_bi.csv')
DEV_DATA_BI = os.path.join(DATADIR, str(UNIGRAM_DIM)+ '_' + str(WINDOW_SIZE) + '_dev_bi.csv')
TEST_DATA_BI = os.path.join(DATADIR, str(UNIGRAM_DIM)+ '_' + str(WINDOW_SIZE) + '_test_bi.csv')

MAX_LEN = 50