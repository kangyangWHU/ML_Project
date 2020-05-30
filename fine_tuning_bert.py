import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,BertModel
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from matplotlib import pyplot as plt

class BertDataset(Dataset):

    def __init__(self, data_path, max_len=200):
        super(BertDataset, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        self.tokens_list, self.atten_mask_list, self.label_list =  self.process_data(data_path, max_len)


    def process_data(self, data_path, max_len):

        data = pd.read_csv(data_path)
        atten_mask_list = []
        tokens_list = []

        for line in data['review'].tolist():

            text_tokens = self.tokenizer.tokenize(line)

            tokens = ["[CLS]"]+ text_tokens + ["[SEP]"]

            if len(tokens) < max_len:
                diff = max_len - len(tokens)
                attn_mask = [1] * len(tokens) + [0] * diff
                tokens += ["[PAD]"] * diff
            else:
                tokens = tokens[:max_len - 1] + ["[SEP]"]
                attn_mask = [1] * max_len

            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            atten_mask_list.append(attn_mask)
            tokens_list.append(tokens_ids)

        return tokens_list, atten_mask_list, data['label'].tolist()

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, index):

        return torch.tensor(self.tokens_list[index]),\
               torch.tensor(self.atten_mask_list[index]),\
               self.label_list[index]


class BertLoader():

    def __init__(self, batch_size, root_path, max_len):

        train_dataset = BertDataset(root_path+"_train.csv", max_len)
        val_dataset = BertDataset(root_path+"_val.csv", max_len)
        test_dataset = BertDataset(root_path+"_test.csv", max_len)
        self._train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self._val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        self._test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    def get_train_loader(self):
        return self._train_loader

    def get_val_loader(self):
        return self._val_loader
    def get_test_loader(self):
        return self._test_loader

class WaiMaiBertModel(nn.Module):
    def __init__(self):
        super(WaiMaiBertModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, atten_mask):

        cont_reps, _  = self.bert(input_ids, attention_mask=atten_mask)
        features = cont_reps[:, 0]
        cls_rep = self.dropout(features)
        outs = self.classifier(cls_rep)
        return outs


def evaluate(model, data_loader, criteon):

    model.eval()
    dev_loss, label_list, pred_list = [], [], []
    for tokens, masks, labels in data_loader:

        tokens = tokens.to(DEVICE)
        masks = masks.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            preds = model(tokens, masks)

        # importance weighting loss
        loss = criteon(preds, labels)

        dev_loss.append(loss.item())
        pred_list.append(preds.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

        torch.cuda.empty_cache()

    pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
    label_list = np.concatenate(label_list, axis=0)

    correct = (pred_list==label_list).sum()
    return np.array(dev_loss).mean(), float(correct)/len(label_list)


def train(args):

    weibo_loader = BertLoader(args.batch_size, ROOT_PATH, args.max_len)

    train_loader = weibo_loader.get_train_loader()
    val_loader = weibo_loader.get_val_loader()

    # construct data loader
    model = WaiMaiBertModel()
    model = model.to(DEVICE)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    criteon = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []


    # log process
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()

        train_loss,label_list,pred_list = [],[],[]
        for tokens, masks, labels in train_loader:

            tokens = tokens.to(DEVICE)
            masks = masks.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero gradient
            optimizer.zero_grad()

            preds = model(tokens, masks)
            loss = criteon(preds, labels)

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_loss.append(loss.item())
            label_list.append(labels.detach().cpu().numpy())
            pred_list.append(preds.detach().cpu().numpy())

            # empty cache
            torch.cuda.empty_cache()

        train_loss = np.array(train_loss).mean()
        val_loss, val_acc = evaluate(model, val_loader, criteon)

        pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
        label_list = np.concatenate(label_list, axis=0)

        correct = (pred_list == label_list).sum()
        train_acc = float(correct) / len(label_list)
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), "best.pt".format(epoch, best_acc))
        torch.save(model.state_dict(), SAVE_PATH + "epoch{}_{:.4f}.pt".format(epoch, val_acc))
        print('Training loss:{}, Val loss:{}'.format(train_loss,val_loss ))
        print("train acc:{:.4f}, val acc:{:4f}".format(train_acc, val_acc))

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

    weibo_loader = BertLoader(args.batch_size, ROOT_PATH, args.max_len)
    test_loader = weibo_loader.get_test_loader()

    # construct data loader
    model = WaiMaiBertModel()
    model.load_state_dict(torch.load(weights_path))
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, test_loader, criterion)
    print("test loss:{:4f}, acc:{:4f}".format(loss, acc))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--clip', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=200)

    ROOT_PATH = 'data/waimai10k/waimai10k'
    SAVE_PATH = 'weights/'
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # th number 3 has the highest priority
    # train(args)
    predict(args, "weights/epoch7_0.9189.pt")
