import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import pickle
import jieba
from collections import Counter
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from matplotlib import pyplot as plt

class WVEmbedding():

    def __init__(self, wv_path, data_path, vocab_size=29000,
                 emb_path=None):

        self.wv_path =wv_path
        self.data_path = data_path
        self.vocab_size = vocab_size

        self.word_list = self.get_word_list()
        self.word_to_id, self.id_to_word = self.get_vocab()
        # load data from saved data, save lots of time
        if emb_path:
            self.embedding = np.load(emb_path)
        else:
            self.embedding = self.get_embedding()

    def get_embedding(self):

        self.wv = KeyedVectors.load_word2vec_format(self.wv_path)
        # get embedding dim
        embedding_dim = self.wv.vector_size
        emb = np.zeros((self.vocab_size, embedding_dim))
        wv_dict = self.wv.vocab.keys()
        num_found = 0
        for idx in tqdm(range(self.vocab_size)):
            word = self.id_to_word[idx]
            if word == '<pad>' or word == '<unk>':
                emb[idx] = np.zeros([embedding_dim])
            elif word in wv_dict:
                emb[idx] = self.wv.get_vector(word)
                num_found += 1

        print("{} of {} found, rate:{:.2f}".format(num_found, self.vocab_size, num_found/self.vocab_size))
        return emb

    # get all words from train data, dev data, test data
    def get_word_list(self):

        data = pd.read_csv(self.data_path, sep=',')
        word_list = []
        for i, line in enumerate(data['review'].values):
            word_list += jieba.lcut(line)
        return word_list

    def get_vocab(self):

        counts = Counter(self.word_list)
        vocab = sorted(counts, key=counts.get, reverse=True)

        # add <pad>
        vocab = ['<pad>', '<unk>'] + vocab

        print('total word size:{}'.format(len(vocab)))
        # trunk vocabulary
        if len(vocab) < self.vocab_size:
            raise Exception('Vocab less than requested!!!')
        else:
            vocab = vocab[:self.vocab_size]

        word_to_id = {word: i for i, word in enumerate(vocab)}
        id_to_word = {i: word for i, word in enumerate(vocab)}

        return word_to_id, id_to_word



class WaiMaiDataSet(Dataset):
    def __init__(self, data_path, word_to_id, max_len=40, use_unk=False):

        self.datas, self.labels = self.load_data(data_path)
        self.max_len = max_len
        self.word_to_id = word_to_id
        self.pad_int = word_to_id['<pad>']

        self.use_unk = use_unk
        # internal data
        self.conversation_list, self.total_len = self.process_data(self.datas)

    def load_data(self, data_path):

        data = pd.read_csv(data_path)
        return data['review'].tolist(), data['label'].tolist()

    # turn sentence to id
    def sent_to_ids(self, text):

        tokens = jieba.lcut(text)
        # if use_unk is True, it will use <unk> vectors
        # else just remove this word
        if self.use_unk:
            token_ids = [self.word_to_id[x] if x in self.word_to_id else self.word_to_id['<unk>'] for x in tokens]
        else:
            token_ids = [self.word_to_id[x] for x in tokens if x in self.word_to_id]

        # Trunking or PADDING
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len]
            text_len = self.max_len
        else:
            text_len = len(token_ids)
            token_ids = token_ids + [self.pad_int] * (self.max_len - len(token_ids))

        return token_ids, text_len

    def process_data(self, data_list):
        conversation_list= []
        total_len = []
        for line in data_list:
            conversation, conver_len = self.sent_to_ids(line)
            conversation_list.append(conversation)
            total_len.append(conver_len)
        return conversation_list, total_len

    def __len__(self):
        return len(self.conversation_list)

    def __getitem__(self, idx):
        return torch.LongTensor(self.conversation_list[idx]),\
               self.total_len[idx], \
               self.labels[idx]


# turn sentence to vector represent,
# average all the word vector as the sentence vector
#
def to_avg_sv(path, save_path, wv_embedding):

    data = pd.read_csv(path)

    sv_list = []
    for line in data['review'].values:
        words = jieba.lcut(line)

        n = 0
        sentence_vector = 0
        for word in words:

            # not <unk>
            try:
                row_index = wv_embedding.word_to_id[word]
                sentence_vector += wv_embedding.embedding[row_index]
                n += 1
            except:
                pass

        # average
        sentence_vector /= n
        sv_list.append(sentence_vector)

    sv = np.array(sv_list)
    np.save(save_path, sv)


def to_concat_sv(path, save_path, wv_embedding, sen_len):

    data = pd.read_csv(path)

    sv_list = []
    for line in data['review'].values:
        words = jieba.lcut(line)

        n = 0
        sentence_vector = []
        for word in words:

            # not <unk>
            try:
                row_index = wv_embedding.word_to_id[word]
                sentence_vector += wv_embedding.embedding[row_index].tolist()
                n += 1
            except:
                pass

        # concat
        if n < sen_len:
            sentence_vector += [0.]*300*(sen_len - n)
        else:

            sentence_vector = sentence_vector[:300 * sen_len]
        sv_list.append(sentence_vector)

    return sv_list
    # sv = np.array(sv_list)
    # np.save(save_path, sv)


def plot_avg_len(path, wv_embedding):

    data = pd.read_csv(path)

    len_list = []
    for line in data['review'].values:
        words = jieba.lcut(line)

        n = 0
        for word in words:

            # not <unk>
            try:
                row_index = wv_embedding.word_to_id[word]
                n += 1
            except:
                pass

        # average
        len_list.append(n)
    plt.hist(len_list,bins=100)
    plt.show()


# spilt single file to train val test file
def split_train_val_test(data_path):
    data = pd.read_csv(data_path, sep=',')
    X = data['review'].tolist()
    Y = data['label'].tolist()
    X_train,X_valtest, y_train, y_valtest = train_test_split(X, Y, test_size=0.2, stratify=Y)

    X_test,X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, stratify=y_valtest)

    # 下面这行代码运行报错
    # list.to_csv('e:/testcsv.csv',encoding='utf-8')
    thead = ['label', 'review']

    def list_to_csv(thead, c1, c2, path):
        data = np.vstack((c1, c2))
        data = np.transpose(data, (1,0))
        df = pd.DataFrame(columns=thead, data=data)  #
        df.to_csv(path, index=False)

    list_to_csv(thead, y_train, X_train, 'weibo100k_train.csv')
    list_to_csv(thead, y_val, X_val, 'weibo100k_val.csv')
    list_to_csv(thead, y_test, X_test, 'weibo100k_test.csv')

    # return X_train,y_train, X_val, y_val, X_test, y_test

def get_data_set(root_path):

    def get_data(mode='train'):
        x_train_path = root_path+"_{}_sv.npy".format(mode)
        y_train_path = root_path+"_{}.csv".format(mode)

        x_train = np.load(x_train_path)
        data = pd.read_csv(y_train_path)
        y_train = data['label'].tolist()
        y_train = np.array(y_train)

        return x_train, y_train

    x_train, y_train = get_data('train')
    x_val, y_val = get_data('val')
    x_test, y_test = get_data('test')

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    wv_path = "D:/datasets/NLP/embedding_cn/sgns.weibo.bigram-char.bz2"
    data_path = "D:/datasets/NLP/waimai10k.csv"
    wv_embedding = WVEmbedding(wv_path, data_path, 29000, emb_path='data/waimai10k/waimai10k_vocab29k_embedding.npy')
    # to_avg_sv("data/waimai10k/waimai10k_train.csv", "data/waimai10k/waimai10k_train_sv.npy", wv_embedding)
    # to_avg_sv("data/waimai10k/waimai10k_val.csv", "data/waimai10k/waimai10k_val_sv.npy", wv_embedding)
    # a = wv_embedding.embedding
    train_data = to_concat_sv("data/waimai10k/waimai10k_train.csv", "data/waimai10k/waimai10k_train_sv.npy", wv_embedding, 100)

    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(train_data)
    print(pca.explained_variance_ratio_)
