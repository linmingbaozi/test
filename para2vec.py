from gensim.models import doc2vec
from collections import namedtuple
import string
import numpy as np

txt = ['./train_img_ann_cat.txt', 'val_img_ann_cat.txt']   # files recording your annotations' path.
save = ['./para2vec_train.npy', './para2vec_val.npy']

for j in range(len(txt)):
    txt_path = txt[j]

    docs = open(txt_path, 'r').readlines()

    for i in range(len(docs)):
        docs[i] = docs[i].split('\t')[1]

    anns = []

    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

    for i, text in enumerate(docs):
        delset = string.punctuation
        text = text.translate(None, delset)
        words = text.lower().split()
        tags = [i]
        anns.append(analyzedDocument(words, tags))

    model =doc2vec.Doc2Vec(anns, size = 100, window = 300, min_count = 1, workers = 4)

    para2vec = np.array(model.docvecs)
    np.save(save[j], para2vec)
