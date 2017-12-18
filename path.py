import numpy as np
txt = ['./val_img_ann_cat.txt', './train_img_ann_cat.txt']
save = ['./wiki_val_path.txt', './wiki_train_path.txt']

for j in range(len(txt)):

    txt_path = txt[j]
    save_path = save[j]

    paths = open(txt_path, 'r').readlines()

    label = ''

    for i in range(len(paths)):
        label += paths[i].split('\t')[0] + '\n'

    print(label)

    with open(save_path, 'w') as f:
        f.write(label)