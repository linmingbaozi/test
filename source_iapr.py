from sklearn.feature_extraction.text import CountVectorizer
from collections import namedtuple
from gensim.models import doc2vec
import scipy.io as sio
import numpy as np
import random
import string
import shutil
import os


def iapr_npy2mat():
	I_tr = np.load('./Iapr_file/iapr_res50_train.npy')
	I_te = np.load('./Iapr_file/iapr_res50_val.npy')
	T_tr = np.load('./Iapr_file/bag_of_word_train.npy')
	T_te = np.load('./Iapr_file/bag_of_word_val.npy')
	L_tr = np.load('./Iapr_file/iapr_cat_train.npy')
	L_te = np.load('./Iapr_file/iapr_cat_val.npy')

	sio.savemant('./Iapr_file/ijcai18_wiki_bag.mat', {'I_tr':I_tr, 'I_te':I_te, 
		'T_tr':T_tr, 'T_te':T_te, 'L_tr':L_tr, 'L_te':L_te})


def iapr_para2vec():
	txt = ['./val_img_ann_cat.txt', './train_img_ann_cat.txt']
	save = ['./para2vec_val.npy', './para2vec_train.npy']

	for j in range(len(txt)):

		txt_path = txt[j]

		docs = open(txt_path,'r').readlines()

		for i in range(len(docs)):
			docs[i] = docs[i].split('\t')[1]
			
		anns = []

		analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

		for i, text in enumerate(docs):
			delset = string.punctuation
			text = text.translate(None, delset) # remove all punctuations existing in the annotations.
			words = text.lower().split()
			tags = [i]
			anns.append(analyzedDocument(words, tags))

		model = doc2vec.Doc2Vec(anns, size=100, window=300, min_count=1,workers=4)

		para2vec =  np.array(model.docvecs)
		np.save(save[j], para2vec)


def iapr_bow():
	txt = ['./train_img_ann_cat.txt', 'val_img_ann_cat.txt']
	save = ['./Iapr_file/bag_of_word_train.npy', './Iapr_file/bag_of_word_val.npy']

	for j in range(len(txt)):
		txt_path = txt[j]

		docs = open(txt_path, 'r').readlines()

		for i in range(len(docs)):
			docs[i] = docs[i].split('\t')[1]
		vectorizer = CountVectorizer(stop_words = 'english', max_features = 1000)
		corpusTotoken = vectorizer.fit_transform(docs).todense()
		feature = np.array(corpusTotoken)
		np.save(save[j], feature)


def iapr_path():

	txt = ['./val_img_ann_cat.txt', './train_img_ann_cat.txt']
	save = ['./coco_val_path.txt', './coco_train_path.txt']

	for j in range(len(txt)):
		txt_path = txt[j]
		save_path = save[j]

		paths = open(txt_path, 'r').readlines()

		label = ''

		for i in range(len(paths)):
			label += paths[i].split('\t')[0] + '\n'


		with open(save_path, 'w') as f:
			f.write(label)

def iapr_label():
	txt = ['./val_img_ann_cat.txt', './train_img_ann_cat.txt']
	save = ['./Iapr_file/iapr_cat_val.npy', './Iapr_file/iapr_cat_train.npy']

	for j in range(len(txt)):

		txt_path = txt[j]
		save_path = save[j]

		paths = open(txt_path, 'r').readlines()

		#label = np.array([[None] * 41])
	    label = np.array([[None] * 1])

		for i in range(len(paths)):
			cat = int(paths[i].split('\t')[2])
			output = np.zeros((1, 1))
			output[0][0] = cat

			# output = np.zeros((1,41))
	        # output[0][cat] = 1

	        label = np.concatenate((label, output))


		label = label[1:]
		np.save(save_path, label)


def iapr_path_anns_cat():
	img_directory = ['./IAPR-TC-12/iaprtc12/train_test/train/', './IAPR-TC-12/iaprtc12/train_test/val/']
	ann_directory = './IAPR-TC-12/iaprtc12/annotations_complete_eng/'
	txt_directory = ['./train_img_ann_cat.txt', './val_img_ann_cat.txt']

	for j in range(len(img_directory)):
		dir = img_directory[j]
		txt = txt_directory[j]
		info = ''
		sub_directory = os.listdir(dir)
		sub_directory.sort()
		for i in range(len(sub_directory)):
			sub = sub_directory[i]
		imgs = os.listdir(os.path.join(dir, sub))
			imgs.sort()
			for img in imgs:
				img_path = os.path.join(dir, sub, img)
				ann_path = os.path.join(ann_directory, sub, img.split('.')[0]+'.eng')

				if not os.path.exists(ann_path):
					os.remove(img_path)
					continue
				idx = i
				
				f = open(ann_path).read()
				desc = f.split("<DESCRIPTION>")[1].split("</DESCRIPTION>")[0]
				desc = desc.replace('\t', ' ')
				desc = desc.replace('\n', ' ')

				print(desc)

				info += img_path + '\t' + desc + '\t' + str(idx) + '\n'

		f = open(txt, 'a')
		f.write(info)


def iapr_split():
	img_dir = os.listdir("./IAPR-TC-12/iaprtc12/images")
	for dir in img_dir:
		os.mkdir("./IAPR-TC-12/iaprtc12/train_test/val/" + dir)
		os.mkdir("./IAPR-TC-12/iaprtc12/train_test/train/" + dir)
		img = os.listdir("./IAPR-TC-12/iaprtc12/images/" + dir)
		img_num = len(img)
		val_num = img_num / 5
		random_idx = range(img_num)
		random.shuffle(random_idx)

		val_idx = random_idx[0:val_num]
		train_idx = random_idx[val_num:]

		for i in val_idx:
			shutil.copy("./IAPR-TC-12/iaprtc12/images/" + dir + '/' + img[i],
                        "./IAPR-TC-12/iaprtc12/train_test/val/" + dir + '/' + img[i])

		for i in train_idx:
			shutil.copy("./IAPR-TC-12/iaprtc12/images/" + dir + '/' + img[i],
                        "./IAPR-TC-12/iaprtc12/train_test/train/" + dir + '/' + img[i])


if __name__ == '__main__':
	iapr_label()