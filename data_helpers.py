import pandas as pd
import numpy as np
from config import WINDOW_SIZE, BATCH_SIZE

class BucketedDataIterator():
    def __init__(self, df, num_buckets=10):
        self.df = df
        self.total = len(df)
        df_sort = df.sort_values('length').reset_index(drop=True)
        self.size = self.total / num_buckets
        self.dfs = []
        for bucket in range(num_buckets - 1):
            self.dfs.append(df_sort.ix[bucket*self.size: (bucket + 1)*self.size - 1])
        self.dfs.append(df_sort.ix[(num_buckets-1)*self.size:])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.pos = 0
        self.shuffle()
        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self):
        if np.any(self.cursor + BATCH_SIZE + 1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i] + BATCH_SIZE - 1]
        
        biwords = list(map(lambda x: list(map(int, x.split(","))), res['biwords'].tolist()))
        words = list(map(lambda x: list(map(int, x.split(","))), res['words'].tolist()))
        tags = list(map(lambda x: list(map(int, x.split(","))), res['tags'].tolist()))

        self.cursor[i] += BATCH_SIZE

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        bi_x = np.zeros([BATCH_SIZE, maxlen + WINDOW_SIZE*2 + 2], dtype=np.int32)
        for i, x_i in enumerate(bi_x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2 + 2] = biwords[i]
        x = np.zeros([BATCH_SIZE, maxlen + WINDOW_SIZE*2 + 1], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2 + 1] = words[i]
        y = np.zeros([BATCH_SIZE, maxlen], dtype=np.int32)
        for i, y_i in enumerate(y):
            y_i[:res['length'].values[i]] = tags[i]
        
        return bi_x, x, y, res['length'].values
    def next_pred_one(self):
        res = self.df.ix[self.pos]
        biwords = list(map(int, res['biwords'].split(',')))
        words = list(map(int, res['words'].split(',')))
        tags = list(map(int, res['tags'].split(',')))
        length = res['length']
        self.pos += 1
        if self.pos == self.total:
            self.pos = 0
        return np.asarray([biwords],dtype=np.int32), np.asarray([words],dtype=np.int32), np.asarray([tags],dtype=np.int32), np.asarray([length],dtype=np.int32)
    def rest_all_batch(self):
        res = self.df.ix[self.pos : ]
        batch_size = self.total - self.pos
        biwords = list(map(lambda x: list(map(int, x.split(","))), res['biwords'].tolist()))
        words = list(map(lambda x: list(map(int, x.split(","))), res['words'].tolist()))
        tags = list(map(lambda x: list(map(int, x.split(","))), res['tags'].tolist()))
              
        maxlen = max(res['length'])
        bi_x = np.zeros([batch_size, maxlen + WINDOW_SIZE*2 + 2], dtype=np.int32)
        for i, x_i in enumerate(bi_x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2 + 2] = biwords[i]
        x = np.zeros([batch_size, maxlen + WINDOW_SIZE*2 + 1], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2 + 1] = words[i]
        y = np.zeros([batch_size, maxlen], dtype=np.int32)
        for i, y_i in enumerate(y):
            y_i[:res['length'].values[i]] = tags[i]

        self.pos = 0
        
        return bi_x, x, y, res['length'].values
    
    def next_all_batch(self):
        res = self.df.ix[self.pos : self.pos + BATCH_SIZE - 1]
        biwords = list(map(lambda x: list(map(int, x.split(","))), res['biwords'].tolist()))
        words = list(map(lambda x: list(map(int, x.split(","))), res['words'].tolist()))
        tags = list(map(lambda x: list(map(int, x.split(","))), res['tags'].tolist()))

        self.pos += BATCH_SIZE
        maxlen = max(res['length'])
        bi_x = np.zeros([BATCH_SIZE, maxlen + WINDOW_SIZE*2 + 2], dtype=np.int32)
        for i, x_i in enumerate(bi_x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2 + 2] = biwords[i]
        x = np.zeros([BATCH_SIZE, maxlen + WINDOW_SIZE*2 + 1], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2 + 1] = words[i]
        y = np.zeros([BATCH_SIZE, maxlen], dtype=np.int32)
        for i, y_i in enumerate(y):
            y_i[:res['length'].values[i]] = tags[i]

        return bi_x, x, y, res['length'].values
       
    def print_info(self):
        print('dfs shape: ', [len(self.dfs[i]) for i in xrange(len(self.dfs))])
        print('size: ', self.size)


