import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import jieba
from utils import WVEmbedding
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib import pyplot as plt
import numpy as np
from torch import optim
from utils import WaiMaiDataSet

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_weights):
        super(TextCNN, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight = nn.Parameter(
            torch.FloatTensor(embedding_weights), requires_grad=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(3, embedding_dim)),
            nn.ReLU(inplace=True)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(4, embedding_dim)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(5, embedding_dim)),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, 2)
        )

    def forward(self, x):

        x = self.embedding_layer(x)
        x = x.unsqueeze(dim=1)

        feature1 = self.conv1(x).squeeze(-1)
        feature2 = self.conv2(x).squeeze(-1)
        feature3 = self.conv3(x).squeeze(-1)

        feature1 = self.max_pool(feature1).squeeze(-1)
        feature2 = self.max_pool(feature2).squeeze(-1)
        feature3 = self.max_pool(feature3).squeeze(-1)

        features = torch.cat([feature1, feature2, feature3], dim=-1)
        x = self.classifier(features)

        return x




def evaluate(model, data_loader, criteon):

    model.eval()
    dev_loss = []
    correct = 0
    for tokens, token_len, labels in data_loader:
        tokens, token_len, labels = tokens.to(DEVICE), token_len.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            preds = model(tokens)
            loss = criteon(preds, labels)
            dev_loss.append(loss.item())

            correct += (preds.argmax(1)==labels).sum()

    return np.array(dev_loss).mean(), float(correct) / len(data_loader.dataset)


def train(args):

    # construct data loader
    train_data_set = WaiMaiDataSet(TRAIN_PATH, wv_embedding.word_to_id, args.max_len, args.use_unk)
    train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)

    dev_data_set = WaiMaiDataSet(DEV_PATH, wv_embedding.word_to_id, args.max_len, args.use_unk)
    dev_data_loader = DataLoader(dev_data_set, batch_size=args.batch_size, shuffle=False)


    model = TextCNN(args.vocab_size, args.embedding_dim, wv_embedding.embedding)
    model = model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    criteon = nn.CrossEntropyLoss().to(DEVICE)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # log process
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = []
        model.train()
        correct = 0
        for tokens, token_len, labels in train_data_loader:
            tokens, token_len, labels = tokens.to(DEVICE), token_len.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(tokens)

            loss = criteon(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            correct += (preds.argmax(1) == labels).sum()
            torch.cuda.empty_cache()

        # learning rate decay
        scheduler.step()

        train_loss = np.array(train_loss).mean()
        train_acc = float(correct) / len(train_data_loader.dataset)
        val_loss, val_acc = evaluate(model, dev_data_loader, criteon)

        # if val_acc> 0.9:
            # best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH+"epoch{}_{:.4f}.pt".format(epoch, val_acc))
        print('epochs:{},Training loss:{:4f}, Val loss:{:4f}'.format(epoch, train_loss,val_loss ))
        print('Training acc:{:4f}, Val acc:{:4f}'.format( train_acc,val_acc ))
        print()


        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    plt.figure()
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.figure()
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'])
    plt.show()


def predict(args, weights_path):

    data_set = WaiMaiDataSet(TEST_PATH, wv_embedding.word_to_id, args.max_len, args.use_unk)
    test_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    model = TextCNN(args.vocab_size, args.embedding_dim, wv_embedding.embedding)

    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(weights_path))
    model = model.to(DEVICE)
    loss, acc = evaluate(model, test_loader, criterion)
    print("test loss:{:4f}, acc:{:4f}".format(loss, acc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # parameters about train model
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--vocab_size', type=int, default=29000)
    parser.add_argument('--use_unk', default=False, action='store_true')
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_PATH = 'data/waimai10k/waimai10k_train.csv'
    DEV_PATH = 'data/waimai10k/waimai10k_val.csv'
    TEST_PATH = 'data/waimai10k/waimai10k_test.csv'
    SAVE_PATH = 'weights/'

    wv_path = "D:/datasets/NLP/embedding_cn/sgns.weibo.bigram-char.bz2"
    data_path = "D:/datasets/NLP/waimai10k.csv"
    emb_path = 'data/waimai10k/waimai10k_vocab29k_embedding.npy'
    wv_embedding = WVEmbedding(wv_path, data_path, 29000, emb_path=emb_path)

    predict(args, "weights/epoch10_0.948.pt")
    # train(args)