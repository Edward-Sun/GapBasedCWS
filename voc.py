#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from config import UNIGRAM_DIM, BIGRAM_DIM, UNIGRAM_PATH, BIGRAM_PATH
from config import DICT_PATH, GOLD_PATH, TEST_PATH, DEST_PATH, Y_PRED, Y_TRUE

class Vocab(object):
    def __init__(self):      
        self.uni2idx = defaultdict(int)
        self.bi2idx = defaultdict(int)
        
        self.uni_vectors = None
        self.bi_vectors = None

        self.load_data()

    def load_data(self):
        with open(UNIGRAM_PATH, 'r') as f:
            line = f.readline().strip().split(" ")
            N, dim = list(map(int, line))

            self.uni_vectors = []
            
            idx = 0
            self.uni2idx['<OOV>'] = idx
            vector = np.zeros(UNIGRAM_DIM, dtype=np.float32)
            self.uni_vectors.append(vector)
            idx += 1
            
            
            for k in range(N):
                line = f.readline().strip().split(" ")
                self.uni2idx[line[0]] = idx
                vector = np.asarray(list(map(float, line[1:])), dtype=np.float32)
                self.uni_vectors.append(vector)
                idx += 1
                
            self.uni_vectors = np.asarray(self.uni_vectors, dtype=np.float32)

        with open(BIGRAM_PATH, 'r') as f:
            line = f.readline().strip().split(" ")
            N, dim = list(map(int, line))
            
            self.bi_vectors = []
            idx = 0
            self.bi2idx['<OOV>'] = idx
            vector = np.zeros(BIGRAM_DIM, dtype=np.float32)
            self.bi_vectors.append(vector)
            idx += 1
            
            for k in range(N):
                line = f.readline().strip().split(" ")
                self.bi2idx[line[0]] = idx
                vector = np.asarray(list(map(float, line[1:])), dtype=np.float32)
                self.bi_vectors.append(vector)
                idx += 1
            
            self.bi_vectors = np.asarray(self.bi_vectors, dtype=np.float32)
            
class Tag(object):
    def __init__(self):
        self.tag2idx = defaultdict(int)
        self.define_tags()

    def define_tags(self):
        self.tag2idx['0'] = 0
        self.tag2idx['1'] = 1

class OOV(object):
    def __init__(self, mydestpath = None):
        self.blank = ' '
        self.mydestpath = mydestpath
        self.dict = defaultdict(int)
        self.word_dict()
        self.ans_segs = self.prod_ans()
        f = open(Y_PRED, "r")
        y_pred = []
        for line in f:
            y_pred += line
        f.close()
        self.prod_pred(self.process_data(), y_pred)


    def word_dict(self):
        f = open(DICT_PATH, 'r')
        li = f.readlines()
        f.close()
        for line in li:
            line = line.strip()
            self.dict[line] = 1

    def prod_ans(self):
        f = open(GOLD_PATH, 'r')
        li = f.readlines()
        f.close()
        ans_segs = []
        for line in li:
            line = line.strip().split(self.blank)
            sent = []
            for word in line:
                sent.append(word)
            ans_segs.append(sent)
        return ans_segs

    def process_data(self):
        src_data = []

        src_data_sentence = []

        f = open(TEST_PATH, 'r')
        li = f.readlines()
        f.close()

        for line in li:
            line = line
            line_t = line.replace('\n', '').replace('\r', '').split('#')
            if (len(line_t) < 3):
                if (len(src_data_sentence) == 0):
                    continue
                src_data.append(src_data_sentence)
                src_data_sentence = []
                continue
            src_word = line_t[0]
            src_data_sentence.append(src_word)

        return src_data

    def prod_pred(self, src_data, seq):
        newSeq = []
        start = 0
        f = open(DEST_PATH, 'w')
        for line in src_data:
            length = len(line)
            end = start + length
            newSeq.append(seq[start:end])
            start = end

        words = ''
        for line, tags in zip(src_data, newSeq):
            for word, label in zip(line, tags):
                words += word
                f.write(word)
                if (label == '1'):
                    f.write(self.blank)
                    words = ''
            f.write('\n')

        f.close()
        return 0


    def prod_pred(self, src_data, tags):
        seg_tags = []
        start = 0
        f = open(DEST_PATH, 'w')
        for line in src_data:
            length = len(line)
            end = start + length
            seg_tags.append(tags[start:end])
            start = end

        for line, tags in zip(src_data, seg_tags):
            for word, label in zip(line, tags):
                f.write(word)
                if (label == '1'):
                    f.write(self.blank)
            f.write('\n')

        f.close()
        return 0


    def eval_oov_rate(self):
        fyt = open(Y_TRUE, 'r')
        yt = fyt.readline().strip()
        fyt.close()
        fyp = open(Y_PRED, 'r')
        yp = fyp.readline().strip()
        fyp.close()
        
        f = open(DEST_PATH, 'r')
        text = f.readlines()
        f.close()
        pred_segs = []
        pred_sent = []
        for line in text:
            line = line.strip().split(self.blank)
            for words in line:
                pred_sent.append(words)
            pred_segs.append(pred_sent)
            pred_sent = []
        
        start = 0
        pointer = 0
        yp_seg = 0
        yt_seg = 0
        y_common = 0
        
        oov_right = 0
        oov_total = 0
        ans_seg = 0
        pred_seg = 0
        common_seg = 0
        for ans_sentence, pred_sentence in zip(self.ans_segs, pred_segs):
            ans = []
            for word in ans_sentence:                
                for i in range(len(word) - 1):
                    ans.append(-1)
                ans.append(word)
            pred = []
            for word in pred_sentence:            
                for i in range(len(word) - 1):
                    pred.append(-1)
                pred.append(word)
                
            ffflag = False
            
            sentence_start = start
            
            for ans_word, pred_word in zip(ans, pred):
                if pred_word != -1:
                    pred_seg+=1
                if ans_word != -1:
                    ans_seg+=1
                if ans_word != -1 and pred_word == ans_word:
                    common_seg+=1
                    chech_flag=True
                else:
                    chech_flag=False
                if yp[pointer] == '1':
                    yp_seg+=1
                if yt[pointer] == '1':
                    yt_seg+=1
                    flag = True
                    for _ in range(start, pointer+1):
                        if yp[_] != yt[_]:
                            flag = False
                    if flag:
                        y_common+=1
                        
                    if flag != chech_flag:
                        
                        if not ffflag:
                            print('|'.join([str(_) for _ in ans]))
                            print('|'.join([str(_) for _ in pred]))
                            print(''.join([str(_) for _ in yt[sentence_start:sentence_start+len(ans)]]))
                            print(''.join([str(_) for _ in yp[sentence_start:sentence_start+len(pred)]]))
                            ffflag = True
                        print(ans_word, pred_word)
                        print(yp[start:pointer+1], yt[start:pointer+1])
                        print('==============')
                    
                    start = pointer
                pointer+=1
                    
                if ans_word != -1 and self.dict.get(ans_word) == None:
                    oov_total += 1
                    if pred_word == ans_word:
                        oov_right += 1
                        
        print(yt_seg, yp_seg, y_common)              
        
        oov_recall_rate = oov_right * 1.0 / oov_total
        print('oov_total=', oov_total, 'oov_right=', oov_right, 'oov_recall_rate=',oov_recall_rate)
        print('ans=', ans_seg, 'pred=', pred_seg, 'common=',common_seg)
        common_seg = common_seg * 1.0
        P, R = common_seg / pred_seg, common_seg / ans_seg
        F = 2/(1/P+1/R)
        print('precision=', P, 'recall=', R, 'F-value=', F)