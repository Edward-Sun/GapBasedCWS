# GapBasedCWS

The implementation of paper *A Gap-Based Framework for Chinese Word Segmentation via Very Deep Convolutional Networks* (https://arxiv.org/abs/1712.09509).

We would like to thank FudanNLP, as we copied and pasted some code from https://github.com/FudanNLP/adversarial-multi-criteria-learning-for-CWS . 

## Dependencies

Python 3.6.3 :: Anaconda custom (64-bit)

Tensorflow: 1.4.1

Numpy: 1.13.3

Pandas: 0.20.3

## Data Format

For **dev, train, test** in each data_directory, its format is:

有#有#1
一#一#0
家#家#1
眼#眼#0
镜#镜#1
店#店#1
销#销#0
售#售#1
的#的#1
３#<NUM>#0
０#<NUM>#0
０#<NUM>#0
度#度#1
老#老#0
花#花#0
镜#镜#1
，#<PUNC>#1

The first one is the original char (，), the second one is the processed char (<PUNC>), the last one is the segmentation tag (1).

## Code Usage

prepare_data_index.py is used produce **.csv** that is used as direct input

model.py & train.py are paired model and train file

## Run

The hyper parameters are defined in config.py and tf.FLAGS

When you have all necessary files:

```bash
python train.py
```