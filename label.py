import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import os
from collections import OrderedDict
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "3"



txt = ['./val_img_ann_cat.txt', './train_img_ann_cat.txt']
save = ['./wiki_cat_val.npy', './wiki_cat_train.npy']

for j in range(len(txt)):

    txt_path = txt[j]
    save_path = save[j]

    paths = open(txt_path, 'r').readlines()

    label = np.array([[None] * 10])

    for i in range(len(paths)):
        cat = int(paths[i].split('\t')[2])


        output = np.zeros((1,10))
        output[0][cat] = 1

        label = np.concatenate((label, output))

    label = label[1:]
    np.save(save_path, label)