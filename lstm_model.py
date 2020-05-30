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

class Attention(nn.Module):
    """
    Computes a weighted average of channels across timesteps (1 parameter pr. channel).
    aka. self-attention
    """

    def __init__(self, attention_size, device=None):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(Attention, self).__init__()
        self.attention_vector = nn.Parameter(torch.FloatTensor(attention_size))
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(attention_size, attention_size)

        nn.init.uniform_(self.attention_vector.data, -0.01, 0.01)

    def forward(self, inputs, input_lengths=None):

        inputs = self.tanh(self.linear(inputs))
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)

        if input_lengths is not None:
            idxes = torch.arange( 0, max_len, device=inputs.device).unsqueeze(0)
            mask = (idxes < input_lengths.unsqueeze(1).to(inputs.device)).float()

            # apply mask and renormalize attention scores (weights)
            masked_weights = unnorm_ai * mask
        else:
            masked_weights = unnorm_ai
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)
        return representations, attentions


class LSTMAttenModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, num_directs, embedding):
        super(LSTMAttenModel, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight = nn.Parameter(torch.FloatTensor(embedding), requires_grad=False)

        bidirectional = True if num_directs == 2 else False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)

        self.attention = Attention(hidden_dim*num_directs)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_dim*num_directs, 2)

    def forward(self, x, x_len):

        embedding = self.embedding_layer(x)

        # pack paded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(embedding, x_len,
                                                         batch_first=True, enforce_sorted=False)
        packed_output, (final_state, _) = self.lstm(packed_input)
        output, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # attention
        features, _ = self.attention(output, unpacked_len)

        features = self.dropout(features)
        logits = self.classifier(features)

        return logits

def evaluate(model, data_loader, criteon):

    model.eval()
    dev_loss = []
    correct = 0
    for tokens, token_len, labels in data_loader:
        tokens, token_len, labels = tokens.to(DEVICE), token_len.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            preds = model(tokens, token_len)
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


    model = LSTMAttenModel(args.embedding_dim, args.hidden_dim, args.vocab_size,
                           args.num_layers, args.num_directs, wv_embedding.embedding)

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
            preds = model(tokens, token_len)

            loss = criteon(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_loss.append(loss.item())
            correct += (preds.argmax(1) == labels).sum()
            torch.cuda.empty_cache()

        # learning rate decay
        scheduler.step()

        train_loss = np.array(train_loss).mean()
        train_acc = float(correct) / len(train_data_loader.dataset)
        val_loss, val_acc = evaluate(model, dev_data_loader, criteon)

        # if val_acc> best_acc:
        #     best_acc = val_acc
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

    model = LSTMAttenModel(args.embedding_dim, args.hidden_dim, args.vocab_size,
                           args.num_layers, args.num_directs, wv_embedding.embedding)

    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(weights_path))
    model = model.to(DEVICE)
    loss, acc = evaluate(model, test_loader, criterion)
    print("test loss:{:4f}, acc:{:4f}".format(loss, acc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # parameters about train model
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=6)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=29000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_directs', type=int, default=2)
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

    predict(args, "weights/epoch10_0.8996.pt")
    # train(args)