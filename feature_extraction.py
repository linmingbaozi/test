import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from collections import OrderedDict
import numpy as np
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
data_transfroms = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()
print(use_gpu)

model_ft = torch.load('resnet50_wiki_lr0.001_momentum0.1_weightdecay0.1_stepsize25_gamma0.1_epoch125.pkl')
new_model = nn.Sequential(
    OrderedDict([
        ('conv1', model_ft.conv1),
        ('bn1', model_ft.bn1),
        ('relu', model_ft.relu),
        ('maxpool', model_ft.maxpool),
        ('layer1', model_ft.layer1),
        ('layer2', model_ft.layer2),
        ('layer3', model_ft.layer3),
        ('layer4', model_ft.layer4),
        ('avgpool', model_ft.avgpool)
    ])
)

txt = ['./val_img_ann_cat.txt', './train_img_ann_cat.txt']
save = ['./res50_val.npy', './res50_train.npy']

for j in range(len(txt)):

    txt_path = txt[j]
    save_path = save[j]

    paths = open(txt_path, 'r').readlines()

    feature = np.array([[None]*2048])

    for i in range(len(paths)):
        path = paths[i].split('\t')[0]
        print(path)
        img = pil_loader(path)
        inputs = data_transfroms(img)

        inputs = inputs.numpy()
        ori_shape = inputs.shape
        inputs = inputs.reshape([1, ori_shape[0],
                                 ori_shape[1],
                                 ori_shape[2]])

        inputs = torch.from_numpy(inputs)

        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        outputs = new_model(inputs).cpu().data.numpy()
        shape = outputs.shape
        new = outputs.reshape((shape[0],shape[1]))
        feature = np.concatenate((feature, new))

    feature = feature[1:]
    np.save(save_path, feature)