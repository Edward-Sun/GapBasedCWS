# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
from config import MAX_LEN, TRAIN_DATA_UNI, DEV_DATA_UNI, TEST_DATA_UNI, TRAIN_PATH, DEV_PATH, TEST_PATH
from config import TRAIN_DATA_BI, DEV_DATA_BI, TEST_DATA_BI, WINDOW_SIZE, GIGA_PATH, BATCH_SIZE

class Data_index(object):
    def __init__(self, Vocabs, Tags):
        self.VOCABS = Vocabs
        self.TAGS = Tags
        self.gen_giga = self.get_giga_lines()
    
    def next_giga(self):
        raw_giga = [next(self.gen_giga) for _ in range(BATCH_SIZE)]
        giga_words = []
        bi_giga_words = []
        virtual_tags = []
        for line in raw_giga:
            words = line.replace('\n', '').replace('\r', '').replace(' ', '#').split('#')
            words = words[1:-1]
            if len(words) > MAX_LEN:
                length = self.rindex(words[:MAX_LEN], virtual_tags)
            else:
                length = len(words)
            bi_idx, word_idx, tag_idx = self.to_bi_index(words[:length], virtual_tags)                       
            
            giga_words.append(word_idx)
            bi_giga_words.append(bi_idx)
        giga_words = np.array(giga_words)
        bi_giga_words = np.array(bi_giga_words)
        
        length = np.array([len(_) for _ in giga_words])
        bi_length = np.array([len(_) for _ in bi_giga_words])
        giga_length = np.array([len(_) - 2 * WINDOW_SIZE - 1 for _ in giga_words])
        
        giga_words = self.numpy_fillna(giga_words, length)
        bi_giga_words = self.numpy_fillna(bi_giga_words, bi_length)
        
        return bi_giga_words.astype(np.int32), giga_words.astype(np.int32), giga_length.astype(np.int32)
        
    def numpy_fillna(self, words, length):
        mask = np.arange(length.max()) < length[:,None]
        out = np.zeros(mask.shape, dtype=words.dtype)
        out[mask] = np.concatenate(words)
        return out


    def get_giga_lines(self):
        while True:
            with open(GIGA_PATH) as giga_file:
                for i in giga_file:
                    yield i
    
    def to_bi_index(self, words, tags):
        word_idx = []
        for _ in range(WINDOW_SIZE):
            words.append('</s>')
            words.insert(0, '</s>')
        for word in words:
            if word in self.VOCABS.uni2idx:
                word_idx.append(self.VOCABS.uni2idx[word])
            else:
                word_idx.append(self.VOCABS.uni2idx['<OOV>'])

        tag_idx = [self.TAGS.tag2idx[tag] for tag in tags]
        
        left = words[:]
        left.insert(0, '</s>')
        right = words[:]
        right.append('</s>')

        bi_idx = []
        
        for current_word, next_word in zip(left, right):
            word = current_word + next_word
            if word in self.VOCABS.bi2idx:
                bi_idx.append(self.VOCABS.bi2idx[word])
            else:
                bi_idx.append(self.VOCABS.bi2idx['<OOV>'])
        
        return bi_idx, word_idx, ','.join(map(str, tag_idx))
    
    def to_index(self, words, tags):
        word_idx = []
        for _ in range(WINDOW_SIZE):
            words.append('</s>')
            words.insert(0, '</s>')
        for word in words:
            if word in self.VOCABS.uni2idx:
                word_idx.append(self.VOCABS.uni2idx[word])
            else:
                word_idx.append(self.VOCABS.uni2idx['<OOV>'])

        tag_idx = [self.TAGS.tag2idx[tag] for tag in tags]
        
        return word_idx, ','.join(map(str, tag_idx))          

    def process_all_data(self):
        train_file_path = TRAIN_DATA_BI
        dev_file_path = DEV_DATA_BI
        test_file_path = TEST_DATA_BI
       
        f_train = open(train_file_path, 'w')
        f_dev = open(dev_file_path, 'w')
        f_test = open(test_file_path, 'w')
        first_row = ['biwords', 'words', 'tags', 'length']
        
        output_train = csv.writer(f_train)
        output_train.writerow(first_row)
        output_dev = csv.writer(f_dev)
        output_dev.writerow(first_row)
        output_test = csv.writer(f_test)
        output_test.writerow(first_row)
        self.process_file(TRAIN_PATH, output_train)
        self.process_file(DEV_PATH, output_dev)
        self.process_file(TEST_PATH, output_test)
        f_train.close()
        f_dev.close()
        f_test.close()
    
    def rindex(self, mylist, mytags):
        if u'<PUNC>' not in mylist or mylist[::-1].index(u'<PUNC>') == len(mylist) - 1:
            if '1' not in mytags:
                return MAX_LEN
            return len(mylist) - mytags[::-1].index('1')
        else:
            return len(mylist) - mylist[::-1].index(u'<PUNC>')
            
    def process_file(self, path, output):
        f = open(path, 'r')
        li = f.readlines()
        f.close()

        data_sentence = []
        label_sentence = []
        
        for line in li:
            line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
            if len(line_t) < 3:
                if len(data_sentence) == 0:
                    continue
                words = data_sentence
                tags = label_sentence
                
                while len(words) > MAX_LEN + 5:
                    length = self.rindex(words[:MAX_LEN], tags[:MAX_LEN])
                    if length == 1:
                        length = MAX_LEN
                    bi_idx, word_idx, tag_idx = self.to_bi_index(words[:length], tags[:length-1])
                    bi_idx = ','.join(map(str, bi_idx))
                    word_idx = ','.join(map(str, word_idx))
                    output.writerow([bi_idx, word_idx, tag_idx, length-1])                        
                              
                    words = words[length:]
                    tags = tags[length:]
                else:
                    length = len(words)
                    if length != 1:
                        bi_idx, word_idx, tag_idx = self.to_bi_index(words[:length], tags[:length-1])
                        bi_idx = ','.join(map(str, bi_idx))
                        word_idx = ','.join(map(str, word_idx))
                        output.writerow([bi_idx, word_idx, tag_idx, length-1])                        
          
                data_sentence = []
                label_sentence = []
                continue
            data_sentence.append(line_t[1])
            label_sentence.append(line_t[2])