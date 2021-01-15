"""
AUBER: Automated BERT-Regularization

Authors:
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.

File: src/utils/utils.py
 - Contain source code for util functions.
"""

import torch
import csv
import sys
import random
import os

def to_tuple(s):
    '''
    coverts a state vector into a tuple
    :param s: state vector
    '''
    s_temp = (s[0] == 0).type(torch.int8).tolist()
    return tuple(s_temp)

def split_train(train_file):

    csv.field_size_limit(sys.maxsize)

    lines = []
    train_lines = []
    folder = '/'.join(train_file.split('/')[:-2])
    new_train = folder + '/dev/train.tsv'
    new_dev = folder + '/train/dev.tsv'

    os.system('rm ' + folder + '/dev/cached*')
    os.system('rm ' + folder + '/train/cached*')

    with open(train_file, 'r') as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        for row in tsv_reader:
            if len(lines) == 0:
                lines.append(row)
                train_lines.append(row)
                continue
            x = random.uniform(0,1)
            if x < 0.25:
                lines.append(row)
            else:
                train_lines.append(row)

    with open(new_dev, 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for l in lines:
            tsv_writer.writerow(l)

    with open(new_train,'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for l in train_lines:
            tsv_writer.writerow(l)
